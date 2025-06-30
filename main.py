from youtube_transcript_fetcher import YouTubeTranscriptFetcher
from transcript_analyzer import TranscriptAnalyzer
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    fetcher = YouTubeTranscriptFetcher()
    url = input('Enter the YouTube video link: ')
    transcript_file = fetcher.fetch_and_save_transcript(url)
    
    if not transcript_file:
        return
    
    analyzer = TranscriptAnalyzer()
    analyzer.load_data(transcript_file)
    
    print("\nPerforming basic analysis...")
    analyzer.basic_analysis()
    
    print("\nAnalyzing common words...")
    analyzer.common_words_analysis()
    
    print("\nPerforming topic modeling...")
    nmf_model, tf_matrix = analyzer.topic_modeling()
    
    print("\nDetecting chapters...")
    chapters = analyzer.detect_chapters(nmf_model, tf_matrix)
    
    print("\nFinal Chapter Points with Names:")
    for time, name in chapters:
        print(f"{time} - {name}")

if __name__ == '__main__':
    main()