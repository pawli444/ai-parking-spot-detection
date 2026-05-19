@echo off
SETLOCAL
set "URL=https://www.youtube.com/watch?v=Pv8N1PamwPQ"
set "OUTPUT=video_dwa.mp4"

where yt-dlp >nul 2>&1
if errorlevel 1 (
    echo yt-dlp not found, installing via pip...
    python -m pip install --user yt-dlp
)

yt-dlp -f best -o "%OUTPUT%" "%URL%"
echo Downloaded to %OUTPUT%
ENDLOCAL