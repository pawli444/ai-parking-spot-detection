param(
    [string]$url = "https://www.youtube.com/watch?v=Pv8N1PamwPQ",
    [string]$output = "video_dwa.mp4"
)

# Ensure yt-dlp is available
if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
    Write-Host "yt-dlp not found, installing via pip..."
    python -m pip install --user yt-dlp
}

# Download best quality and save as the specified output
yt-dlp -f best -o $output $url
Write-Host "Downloaded to $output"