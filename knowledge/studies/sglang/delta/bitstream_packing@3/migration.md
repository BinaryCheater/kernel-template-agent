# Migration from bitstream_packing@2

Keep code_bits,word_bits,length/range and all bit-coordinate/extraction fields unchanged. The bit_identity and memory_safety obligations are unchanged.

The old group_ownership wording named only complete32-code groups despite version2's byte-stream scope. Version3 makes this ownership choice executable. Old full32-code writers bind ownership_unit=complete32_codes; its conditional group obligation preserves the old statement and its contained containers have one owner. Existing byte-stream bindings may bind whole_containers only after their actual exclusive container ownership and zero initialization are shown; version2 did not itself discharge those native obligations. These facts now have named pre_writer ownership and initialization premises.

The mathematical bitstream domain remains whole containers of1..8-bit codes. The newly explicit native writer partition admits padded8/16-bit single-owner mask rows without pretending multiple row writers form one32-code owner. Old versions and source instances remain unchanged. The ownership proof must be replayed per realization; bit injection alone cannot prove race freedom.
