#!/bin/bash
set -e

python scripts/compute_norm_stats.py --config-name pi05_yam_sortitem

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_yam_sortitem --exp-name pi05_yam_sortitem_abs_v0
