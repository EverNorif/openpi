#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=7 uv run scripts/compute_norm_stats.py --config-name pi05_yam_sortitem

CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_yam_sortitem --exp-name pi05_yam_sortitem_abs_v0
