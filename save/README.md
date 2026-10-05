# FLORA save directory

This directory stores generated FLORA artifacts. 

## results/

Final alignment outputs, Turtle files (`.ttl`) written by `src/main.py`, for example `my_dataset.ttl`.

## logs/

Run logs from `src/main.py`, for example `log_my_dataset.txt`.

## checkpoints/

Optional resumable checkpoints created when `--enable_checkpoint` is used.
Checkpoints are grouped by output stem and named like
`checkpoint_iter_0001.pkl`.

## cache/

Reusable intermediate results that speeds up repeated runs on the same dataset. This includes loaded knowledge graphs, compact array-based graph representations, predicate
functionalities, and literal matching results.
