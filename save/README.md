# FLORA save directory

This directory stores generated FLORA artifacts. 

## results/

Final alignment outputs, usually Turtle files (`.ttl`) written by `src/main.py`, for example `dw-v2.txt`.

## logs/

Run logs from `src/main.py`. FLORA names the main log from the output file stem, for example `log_dw-v2.txt`.

## checkpoints/

Optional resumable checkpoints created when `--enable_checkpoint` is used.
Checkpoints are grouped by output stem and named like
`checkpoint_iter_0001.pkl`.

## cache/

Reusable intermediate results that speeds up repeated runs. This includes parsed knowledge graph caches, compact mmap graph representations, predicate
functionalities, and literal matching results.
