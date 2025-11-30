import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import requests

def main(bookmark: dict) -> dict:
    """
    Custom parser for YouTube URLs.

    Detects YouTube URLs (including youtu.be), fetches video metadata including
    title, channel name, description, and transcript (preferring manual over auto-generated).

    Parameters:
        bookmark (dict): Bookmark dictionary with 'url', 'title', etc.

    Returns:
        dict: Updated bookmark dictionary or original if not YouTube.
    """
    url = bookmark.get('url', '')
    if not url:
        return bookmark

    # Check if it's a YouTube URL
    youtube_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.match(youtube_pattern, url)
    if not match:
        return bookmark  # Not a YouTube URL, return unchanged

    video_id = match.group(1)

    try:
        # Fetch video metadata using YouTube API (oEmbed)
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        response.raise_for_status()
        metadata = response.json()

        title = metadata.get('title', bookmark.get('title', 'Unknown Title'))
        author_name = metadata.get('author_name', 'Unknown Channel')

        # Update title to "Video title, by Channel name"
        bookmark['title'] = f"{title}, by {author_name}"

        # Fetch transcript
        try:
            # Use the simpler API which automatically tries English first
            # then falls back to other available languages
            api = YouTubeTranscriptApi()
            print(f"Attempting to fetch transcript for video {video_id}...")
            transcript_data = api.fetch(video_id)
            print(f"Fetched transcript data: {type(transcript_data)}, length: {len(transcript_data) if transcript_data else 0}")
            
            if transcript_data:
                formatter = TextFormatter()
                transcript_text = formatter.format_transcript(transcript_data)
                print(f"Formatted transcript text length: {len(transcript_text)}")
                # Combine description and transcript
                description = metadata.get('description', '')
                bookmark['description'] = f"{description}\n\n{transcript_text}" if description else transcript_text
            else:
                # No transcript available, just use description
                print(f"No subtitles found for video {video_id} (transcript_data is empty)")
                bookmark['description'] = metadata.get('description', '')

        except Exception as e:
            print(f"Failed to fetch transcript for video {video_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to just description
            bookmark['description'] = metadata.get('description', '')

    except Exception as e:
        print(f"Failed to fetch YouTube metadata for video {video_id}: {e}")
        # Return original bookmark if metadata fetch fails
        return bookmark

    return bookmark