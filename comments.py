import yt_dlp

video_url = "https://www.youtube.com/shorts/kFAs3nZoKps"
ydl_opts = {
    "getcomments": True,
}

def get_title(url):
    """ Extract subtitles for a given video URL """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Unknown Title")
    
    print(f"Video Title: {title}\n")

def get_comments(url):
    """ Extract comments from a given video URL """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        comments = info.get("comments", [])
    
    for comment in comments[:10]:  # Limit to 10 comments for readability
        print(f"{comment['author']}: {comment['text']}")

# Call functions
get_title(video_url)
get_comments(video_url)
