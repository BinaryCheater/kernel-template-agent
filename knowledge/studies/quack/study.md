# quack NVIDIA source study

Question: which reusable refinement choices explain every materially different computation organization, layout, pipeline, quantization and schedule in this pinned library?

Assignment: Worker C, QuACK → apex → Mamba → causal-conv1d, as in knowledge/studies/assignments.md. Only assigned study directories are written. No commits or shared-index edits. Stop after causal-conv1d.

Source: https://github.com/Dao-AILab/quack, revision `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8`. Full retained superproject tree and archive: [manifest](inputs/sources.json); [full file listing](inputs/source-files.txt). External submodules are not used as implementation evidence. Upstream LICENSE files remain in the retained tree.

Input pool: repository seed and empty reviewed index at initial read; [exact version list](inputs/pool.json) and copied inputs/pool-records preserve all referenced records. External study proposals are provisional.

Status: complete source extraction after continuation. The initial pass closed prematurely; its useful records are preserved in *-initial.md. Current [inventory](inventory.md), [concrete design ledger](designs.md), [delta index](delta/README.md), and [review menu](review.md) supersede tentative initial matches. 11 candidate versions have paired outlines. Native realization and numeric assurance obligations are identified; materially different designs are not left unread.

Validation: [focused algebra/index replays and schema checks](checks/designs.md). No GPU experiments, WP gate or commits are part of this source study.
