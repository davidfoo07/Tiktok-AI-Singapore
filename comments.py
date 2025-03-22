import yt_dlp

video_url = "https://www.youtube.com/shorts/kFAs3nZoKps"
ydl_opts = {'format': 'best', 'outtmpl': "temp.mp4", 'nooverwrites': False, }

def get_title(url):
    """ Extract subtitles for a given video URL """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)
        title = info.get("title", "Unknown Title")
    
    print(f"Video Title: {title}\n")

# Call functions
get_title(video_url)