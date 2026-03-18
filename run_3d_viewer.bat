@echo off
REM FIFA 3D Viewer Launcher for Windows
REM Usage: run_3d_viewer.bat [sequence_name] [frame_number]
REM Example: run_3d_viewer.bat ARG_CRO_225412 0

set SEQUENCE=%1
set FRAME=%2

if "%SEQUENCE%"=="" set SEQUENCE=ARG_CRO_225412
if "%FRAME%"=="" set FRAME=0

echo Starting FIFA 3D Visualization...
echo Sequence: %SEQUENCE%
echo Frame: %FRAME%
echo.
echo Controls:
echo   - Mouse drag: Rotate view
echo   - Mouse wheel: Zoom in/out
echo   - Space: Play/Pause
echo   - Left/Right arrows: Previous/Next frame
echo   - Q: Quit
echo.

python visualize_3d_interactive.py --sequence %SEQUENCE% --frame %FRAME%
