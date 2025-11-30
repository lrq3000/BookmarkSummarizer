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
        # Create a session with a browser-like User-Agent to avoid being blocked by YouTube
        # This is crucial as YouTube often blocks requests from default library User-Agents (like python-requests)
        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9"
            })

            # Fetch video metadata using YouTube API (oEmbed)
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = session.get(oembed_url, timeout=10)
            response.raise_for_status()
            metadata = response.json()

            title = metadata.get('title', bookmark.get('title', 'Unknown Title'))
            author_name = metadata.get('author_name', 'Unknown Channel')

            # Update title to "Video title, by Channel name"
            bookmark['title'] = f"{title}, by {author_name}"

            # Fetch transcript
            try:
                # Try to get manual transcript first, fallback to auto-generated
                # Pass the session with custom User-Agent to the API to prevent "YouTube is blocking requests from your IP" errors
                api = YouTubeTranscriptApi(http_client=session)
                transcript_list = api.list(video_id)
                transcript = None

                # Prefer manual transcript
                for t in transcript_list:
                    if t.is_generated:
                        continue  # Skip auto-generated
                    try:
                        transcript = t.fetch()
                        break
                    except Exception:
                        continue

                # If no manual transcript, try auto-generated
                if transcript is None:
                    for t in transcript_list:
                        if t.is_generated:
                            try:
                                transcript = t.fetch()
                                break
                            except Exception:
                                continue

                if transcript:
                    formatter = TextFormatter()
                    transcript_text = formatter.format_transcript(transcript)
                    # Combine description and transcript
                    description = metadata.get('description', '')
                    bookmark['description'] = f"{description}\n\n{transcript_text}"
                else:
                    # No transcript available, just use description
                    print(f"No subtitles found for video {video_id}")
                    bookmark['description'] = metadata.get('description', '')

            except Exception as e:
                print(f"Failed to fetch transcript for video {video_id}: {e}")
                # Fallback to just description
                bookmark['description'] = metadata.get('description', '')

    except Exception as e:
        print(f"Failed to fetch YouTube metadata for video {video_id}: {e}")
        # Return original bookmark if metadata fetch fails
        return bookmark

    return bookmark
