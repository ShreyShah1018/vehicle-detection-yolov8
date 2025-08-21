import yt_dlp
import cv2

def get_youtube_stream_url(youtube_url: str) -> str:
    """Get direct video stream URL from YouTube Live."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best[ext=mp4]',
        'forceurl': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def open_youtube_stream(youtube_url: str):
    stream_url = get_youtube_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)
    return cap
