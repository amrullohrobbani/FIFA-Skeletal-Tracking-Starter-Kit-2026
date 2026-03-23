"""
FIFA 3D Skeleton Visualization Launcher
Quick-start script for the 3D interactive viewer
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_3d_interactive import main

if __name__ == '__main__':
    # Default configuration - modify these as needed
    default_args = [
        '--sequence', 'ARG_CRO_225412',  # Default sequence
        '--frame', '0',  # Starting frame
    ]
    
    # If user provides arguments, use those instead
    if len(sys.argv) > 1:
        main()
    else:
        # Use default arguments
        sys.argv.extend(default_args)
        main()
