"""Dynamic dependencies of an ELF module: DT_NEEDED entries and undefined dynamic symbols.

Pure Python over the module bytes, so it runs on the control plane without tools. A
cubin has no dynamic section; a shared library lists what it links.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from kta.core.errors import CheckerError

SHT_DYNAMIC = 6
SHT_DYNSYM = 11
DT_NEEDED = 1
ELFCLASS64 = 2


@dataclass(frozen=True)
class DynamicInfo:
    needed: list[str] = field(default_factory=lambda: list[str]())
    undefined: list[str] = field(default_factory=lambda: list[str]())


def _cstr(table: bytes, offset: int) -> str:
    end = table.index(b"\0", offset)
    return table[offset:end].decode("utf-8", "replace")


def dynamic_info(data: bytes) -> DynamicInfo:
    if data[:4] != b"\x7fELF":
        raise CheckerError("module is not an ELF file")
    if data[4] != ELFCLASS64 or data[5] != 1:
        raise CheckerError("only little-endian ELF64 modules are supported")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)
    ]
    needed: list[str] = []
    undefined: list[str] = []
    for _name, stype, _f, _a, off, size, link, _i, _al, entsize in sections:
        if stype not in (SHT_DYNAMIC, SHT_DYNSYM) or entsize == 0:
            continue
        strtab = sections[link]
        strings = data[strtab[4] : strtab[4] + strtab[5]]
        for pos in range(off, off + size, entsize):
            if stype == SHT_DYNAMIC:
                tag, val = struct.unpack_from("<qQ", data, pos)
                if tag == DT_NEEDED:
                    needed.append(_cstr(strings, val))
            else:
                st_name, _info, _other, shndx, _value, _size = struct.unpack_from(
                    "<IBBHQQ", data, pos
                )
                if shndx == 0 and st_name:
                    undefined.append(_cstr(strings, st_name))
    return DynamicInfo(needed, undefined)
