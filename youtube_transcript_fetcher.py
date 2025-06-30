import re
import csv
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os

load_dotenv()

class YouTubeTranscriptFetcher:
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')

    def get_video_id(self, url):
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        return video_id_match.group(1) if video_id_match else None

    def get_video_title(self, video_id):
        youtube = build('youtube', 'v3', developerKey=self.api_key)

        request = youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()

        title = response['items'][0]['snippet']['title'] if response['items'] else 'Unknown Title'
        return title

    def get_video_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def save_to_csv(self, title, transcript, filename):
        transcript_data = [{'start': entry['start'], 'text': entry['text']} for entry in transcript]
        df = pd.DataFrame(transcript_data)
        df.to_csv(filename, index=False)

        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Title:', title])

    def fetch_and_save_transcript(self, url, output_filename=None):
        video_id = self.get_video_id(url)

        if not video_id:
            print('Invalid YouTube URL.')
            return None

        title = self.get_video_title(video_id)
        transcript = self.get_video_transcript(video_id)

        if not transcript:
            print('No transcript available for this video.')
            return None

        filename = output_filename if output_filename else f"{video_id}_transcript.csv"
        self.save_to_csv(title, transcript, filename)
        print(f'Transcript saved to {filename}')
        return filename