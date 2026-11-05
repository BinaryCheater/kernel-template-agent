"""The pilot's families as `mip.Family`: factors, walls and the precompile register estimate."""

import itertools

from tools.kta_bridge import domain, forms
from tools.kta_bridge.mip import Family


def hgemm(cache: str = "warm") -> Family:
    menu = (64, 96, 128, 192, 256)

    def assign(a, b, t):
        return {"BM": a[0], "BN": a[1], "BK": b[0], "s": b[1], "T": t}

    def smem(a, b):
        return domain.smem_of("hgemm", assign(a, b, 128))

    def regs(a, b, t):
        return domain.regs_estimate("hgemm", assign(a, b, t))

    def amounts(a, b, t):
        x = assign(a, b, t)
        return forms.hgemm(domain.HGEMM_SHAPE, x, regs(a, b, t), smem(a, b), cache)

    return Family(
        a=tuple(itertools.product(menu, menu)),
        b=tuple(itertools.product((16, 32, 48, 64, 96), (1, 2, 3, 4))),
        t=(64, 128, 256),
        amounts=amounts,
        smem=smem,
        regs=regs,
        assign=assign,
        width=lambda b: b[0],
        stage=lambda b: b[1],
    )


def fa(cache: str = "warm") -> Family:
    menu = (32, 48, 64, 96, 128)

    def assign(a, b, t):
        return {"BM": a[0], "BN": a[1], "s": b, "T": t}

    def smem(a, b):
        return domain.smem_of("fa_forward", assign(a, b, 128))

    def regs(a, b, t):
        return domain.regs_estimate("fa_forward", assign(a, b, t))

    def amounts(a, b, t):
        x = assign(a, b, t)
        return forms.fa(domain.FA_SHAPE, x, regs(a, b, t), smem(a, b), cache)

    return Family(
        a=tuple(itertools.product(menu, menu)),
        b=(1, 2, 3),
        t=(64, 128, 256),
        amounts=amounts,
        smem=smem,
        regs=regs,
        assign=assign,
        regs_support_b=True,
    )


MOE_SHAPE = {"E": 6, "T": 5760, "N": 1536, "K": 2048}


def moe_smem(x: dict) -> int:
    per_trip = x["BM"] * x["BK"] * 2 + x["BN"] * x["BK"] // 2 + x["BN"] * x["BK"] // 8
    return x["s"] * per_trip + x["BN"] * x["BK"] * 2


def moe_nvfp4(cache: str = "warm") -> Family:
    bms = (32, 48, 64, 80, 96, 120, 160, 192, 240, 320)  # divide the 960 tokens of an expert
    bns = (32, 48, 64, 80, 96, 128, 160, 192, 256, 320)

    def assign(a, b, t):
        return {"BM": a[0], "BN": a[1], "BK": b[0], "s": b[1], "T": t}

    def smem(a, b):
        return moe_smem(assign(a, b, 128))

    def regs(a, b, t):
        return min(255, int(a[0] * a[1] / t + 62))

    def amounts(a, b, t):
        return forms.moe_nvfp4(MOE_SHAPE, assign(a, b, t), regs(a, b, t), smem(a, b), cache)

    return Family(
        a=tuple(itertools.product(bms, bns)),
        b=tuple(itertools.product((16, 32, 48, 64, 96), (1, 2, 3, 4))),
        t=(64, 128, 256, 512),
        amounts=amounts,
        smem=smem,
        regs=regs,
        assign=assign,
        width=lambda b: b[0],
        stage=lambda b: b[1],
    )
