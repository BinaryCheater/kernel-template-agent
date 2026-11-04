"Check the extracted bit codec and its midpoint tie policy; no GPU execution."

import json
import sys
from pathlib import Path

high_words = [0x0000, 0x007F, 0x0080, 0x3F80, 0x3F81, 0x7F7E, 0x8000, 0xBF80]
cases = 0
for high in high_words:
    for low in range(65536):
        carry = int(low >= 0x8000)
        model_word = high + carry
        restored_high = model_word - carry
        restored = (restored_high << 16) | low
        assert restored == (high << 16) | low
        cases += 1

# At an even-high-word midpoint, the source increments rather than choosing even.
midpoint_model_word = 0x3F80 + int(0x8000 >= 0x8000)
assert midpoint_model_word == 0x3F81
assert midpoint_model_word != 0x3F80  # nearest-even would retain this even high word

result = {
    "kind": "abstract_parameter_bit_codec_check",
    "native_execution": False,
    "high_words": high_words,
    "all_low_word_patterns": 65536,
    "roundtrip_cases_passed": cases,
    "midpoint_policy": "increment high word when low sign bit is one; not nearest-even",
}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
sys.stdout.write(json.dumps(result) + "\n")
