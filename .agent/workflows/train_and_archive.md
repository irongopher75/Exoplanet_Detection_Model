---
description: Run training loop and archive processed data
---

This workflow runs the training script and automatically moves the processed data files to an archive folder upon completion. This is useful for continuous training pipelines where you want to "consume" data from a queue.

### Prerequisite
Ensure you have processed data in `data/processed` (or your download folder).

### Steps

1. Run the training script with the `--archive-dir` argument.

```bash
# Mac/Linux
python3 scripts/run_training.py \
    --data-dir data/processed \
    --archive-dir data/archive \
    --epochs 20 \
    --output-dir outputs

# Windows
python scripts\run_training.py ^
    --data-dir data\processed ^
    --archive-dir data\archive ^
    --epochs 20 ^
    --output-dir outputs
```

2. Verify that files have been moved from `data/processed` to `data/archive`.

**Note:** If a file with the same name exists in the archive, it will be renamed with a timestamp suffix to prevent overwriting.
