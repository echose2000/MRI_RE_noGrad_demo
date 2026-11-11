# Project Name

Demo code

---

## 📦 Environment Setup

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# create and sync environment
uv sync

# activate shell (optional)
uv shell
```

--- 
## Run the demo

```bash
uv run python test_2D.py --data /path/to/sample.h5 --recursive-steps 50
```

`--recursive-steps 50` runs 50 model calls per training iteration. The loss
is computed on the final output and backpropagated through all 50 calls.
Evaluation reports calls 1–49 as `stage_1` through `stage_49` and call 50 as `final`.
