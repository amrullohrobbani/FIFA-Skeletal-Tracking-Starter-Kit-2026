@echo off
REM FIFA 3D Skeleton Visualization Launcher
REM Double-click this file to start the 3D viewer

echo Starting FIFA 3D Skeleton Viewer...
echo.

REM You can change the sequence and frame here:
set SEQUENCE=ARG_CRO_225412
set FRAME=0

REM Activate conda and run visualization
call conda activate base
python visualize_3d_interactive.py --sequence %SEQUENCE% --frame %FRAME%

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit...
    pause > nul
)
