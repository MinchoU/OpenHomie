#!/bin/bash
# Wrapper that runs the visualizer in a completely clean environment.
# Usage: bash run_debug_vis.sh /tmp/debug_obs.pt [--port 8090]

SCRIPT="$(cd "$(dirname "$0")" && pwd)/legged_gym/legged_gym/utils/debug_obs_visualizer.py"

exec /usr/bin/env -i \
    HOME="$HOME" \
    TERM="$TERM" \
    PYTHONNOUSERSITE=1 \
    /scratch/cmw9903/envs/miniconda3/envs/homierl/bin/python -s "$SCRIPT" "$@"
