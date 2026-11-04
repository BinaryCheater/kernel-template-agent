# Batch r9c: entry block

r9b again, same hypotheses (`../r9/entry.md`): its rotating points were lost to a flush kernel past the end of its buffer (cuda error 700). Reason for the repeat: a harness fault, the warm points of r9b stand. New point: `seq` at 2 blocks to a tile (Q32: warm at two blocks per SM runs 1766 B/clk for the chip, 24 tiles, replay about twice as slow).
