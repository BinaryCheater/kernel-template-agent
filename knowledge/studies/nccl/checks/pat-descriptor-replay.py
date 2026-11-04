"Compile only retained host/device PAT descriptor algorithms as short CPU source replay."

import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (
    Path(json.loads((root / "inputs/sources.json").read_text())["tree"])
    / "src/include/collectives.h"
)
body = source.read_text()
classes = []
for name in ["PatRSAlgorithm", "PatAGAlgorithm"]:
    start = body.index("template <typename T>\nclass " + name)
    end = body.index("\n};", start) + 4
    classes.append(body[start:end])
preamble = (
    "#include <cstddef>\n#include <cstdint>\n#include <cstdio>\n#include "
    "<cassert>\n#include <sys/types.h>\n#define __device__\n#define __hos"
    "t__\n#define COMPILER_FFS __builtin_ffs\n#define COMPILER_POPCOUNT3"
    "2 __builtin_popcount\nint log2Up(unsigned x){return x<=1?0:32-__bu"
    "iltin_clz(x-1);}\nstatic constexpr int PatUsed=1,PatSkipped=2;\nstr"
    "uct ncclPatStep {int recvDim,sendDim,recvOffset,sendOffset,stepOf"
    "fset,postRecv,postSend,nelem,last,flags;int step,nvlsOffset;size_"
    "t inpIx,outIx;};\n"
)
main = (
    "\ntemplate<template<typename> class Algo> int replay(int P,int r,i"
    "nt size,int depth){\n Algo<float> a(size,depth,16,0,9,9,4,r,P);int"
    " G=a.getParallelFactor();assert(G>0 && G<=16 && (G&(G-1))==0);\n b"
    "ool terminal[16]={};int i=0,skips=0;\n for(;i<100000;i++) {ncclPat"
    "Step s{};a.getNextOp(&s);assert(s.nelem>=0 && s.nelem<=4);assert("
    "s.nvlsOffset>=0 && s.nvlsOffset<G*s.nelem);\n if(s.flags&PatSkippe"
    "d)skips++;else{assert(s.sendDim>=-1 && s.sendDim<log2Up(P));asser"
    "t(s.recvDim>=-1 && s.recvDim<log2Up(P));assert(s.inpIx<size_t(P*9"
    "));assert(s.outIx<size_t(P*9));}\n if(s.last)terminal[i%G]=true;\n "
    "if(s.last==2)break;\n }\n assert(i<100000);for(int g=0;g<G;g++)asse"
    "rt(terminal[g]);return i+1;\n}\nint main(){int cases=0,steps=0;for("
    "int P=2;P<=17;P++)for(int r=0;r<P;r++)for(int size: {16,4096})for"
    "(int depth: {1,8}){steps+=replay<PatRSAlgorithm>(P,r,size,depth);"
    'steps+=replay<PatAGAlgorithm>(P,r,size,depth);cases+=2;}printf("%'
    'd %d\\n",cases,steps);}\n'
)
# initializer-list loops need their standard header.
with tempfile.TemporaryDirectory(prefix="kta-pat-source-replay-") as temp:
    src = Path(temp) / "pat.cpp"
    exe = Path(temp) / "pat"
    src.write_text(preamble + "\n#include <initializer_list>\n" + "\n".join(classes) + main)
    subprocess.run(
        ["c++", "-std=c++17", "-O0", str(src), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    cases, steps = map(int, completed.stdout.split())
(root / "checks/pat-descriptor-result.json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "cases": cases,
            "descriptor_steps": steps,
            "checks": [
                "finite terminal schedule",
                "all stride-assigned groups receive termination",
                "legal live send/recv dimension",
                "live input/output index bounds",
                "NVLS offset and element count",
            ],
            "source": (
                "src/include/collectives.h PatRSAlgorithm/PatAGAlgorithm retained host methods"
            ),
            "scope": (
                "CPU descriptor generation only; not collective values, worker syn"
                "chronization, GPU or network execution"
            ),
        },
        indent=2,
    )
    + "\n"
)
sys.stdout.write(" ".join(str(value) for value in [cases, "PAT source cases passed"]) + "\n")
