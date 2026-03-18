# FIFA 3D Skeleton Visualization Launcher (PowerShell)
# Double-click this file to start the 3D viewer

Write-Host "Starting FIFA 3D Skeleton Viewer..." -ForegroundColor Green
Write-Host ""

# Default configuration
$SEQUENCE = "ARG_CRO_225412"
$FRAME = 0

# Read config file if it exists
if (Test-Path "viewer_config.txt") {
    Write-Host "Reading configuration from viewer_config.txt..." -ForegroundColor Cyan
    Get-Content "viewer_config.txt" | ForEach-Object {
        if ($_ -match '^SEQUENCE=(.+)$') {
            $SEQUENCE = $matches[1]
        }
        if ($_ -match '^FRAME=(\d+)$') {
            $FRAME = $matches[1]
        }
    }
}

Write-Host "Sequence: $SEQUENCE" -ForegroundColor Yellow
Write-Host "Starting Frame: $FRAME" -ForegroundColor Yellow
Write-Host ""

# Run visualization
python visualize_3d_interactive.py --sequence $SEQUENCE --frame $FRAME

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error occurred. Press any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
