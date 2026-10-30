"""WP5 test data: emits only the step markers of the chain (not a kernel)."""

KEYS = ["k_split", "ab_smem", "ab_async", "ab_stages", "ab_swizzle"]


def generate(assignment, case):
    body = []
    for key in KEYS:
        body += [f"// kta:step {key} begin", f"// {key}: {assignment}", f"// kta:step {key} end"]
    return {"files": {"kernel.cu": "\n".join(body)}, "markers": KEYS}
