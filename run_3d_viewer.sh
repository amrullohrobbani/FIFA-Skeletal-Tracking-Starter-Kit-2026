#!/bin/bash
# FIFA 3D Viewer Launcher for Linux
# Usage: ./run_3d_viewer.sh [sequence_name] [frame_number]
# Example: ./run_3d_viewer.sh ARG_CRO_225412 0

SEQUENCE=${1:-ARG_CRO_225412}
FRAME=${2:-0}

echo "Starting FIFA 3D Visualization..."
echo "Sequence: $SEQUENCE"
echo "Frame: $FRAME"
echo ""
echo "Controls:"
echo "  - Mouse drag: Rotate view"
echo "  - Mouse wheel: Zoom in/out"
echo "  - Space: Play/Pause"
echo "  - Left/Right arrows: Previous/Next frame"
echo "  - Q: Quit"
echo ""

python visualize_3d_interactive.py --sequence "$SEQUENCE" --frame "$FRAME"
