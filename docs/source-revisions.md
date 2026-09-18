# Source revision aliases

Historical benchmark CSVs retain their original `source_rev` and
`timestamp_utc` values. After the repository history was reorganized,
`source-revisions.csv` maps those recorded identifiers to source snapshots
in the current history. Commit dates in the reorganized history are assigned
dates; the CSV timestamps remain the measurement record.

For `tracked_source_identical`, the tracked files under `csrc/`,
`triton_kernels/`, `scripts/`, `Makefile`, `requirements.txt` and
`requirements-lock.txt` are byte-for-byte identical to the recorded source
revision. Check out `source_commit` to inspect or build that source snapshot.
Results and analysis documents are published later in the history and need
not exist in the source snapshot itself.

`base_only_dirty_run` preserves an existing `-dirty` qualifier: the alias
identifies the committed base, not the uncommitted changes used for that run.
It does not make that run reproducible or promote its evidence status.

These aliases describe source equivalence, not the original wall-clock
development sequence. Original measurements and performance conclusions are
unchanged.
