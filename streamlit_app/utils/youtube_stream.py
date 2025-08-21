import yt_dlp
import cv2

def get_youtube_stream_url(youtube_url: str, max_height: int = 720) -> str:
    """Get direct video stream URL from YouTube Live with max height constraint."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': f'best[height<={max_height}][ext=mp4]/best[ext=mp4]',
        'forceurl': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def open_youtube_stream(youtube_url: str, max_height: int = 720):
    """Open YouTube stream with specified max height."""
    stream_url = get_youtube_stream_url(youtube_url, max_height)
    cap = cv2.VideoCapture(stream_url)
    return cap
