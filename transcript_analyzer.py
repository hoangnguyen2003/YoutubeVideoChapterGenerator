import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

class TranscriptAnalyzer:
    def __init__(self):
        self.df = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df['start'] = pd.to_numeric(self.df['start'], errors='coerce')
        return self.df

    def basic_analysis(self):
        print("Dataset Overview:")
        print(self.df.info())
        print("\nBasic Statistics:")
        print(self.df.describe())

        self.df['text_length'] = self.df['text'].apply(len)
        plt.figure(figsize=(10, 5))
        plt.hist(self.df['text_length'], bins=50, color='blue', alpha=0.7)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.show()

    def common_words_analysis(self):
        vectorizer = CountVectorizer(stop_words='english')
        word_counts = vectorizer.fit_transform(self.df['text'])
        word_counts_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
        common_words = word_counts_df.sum().sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(10, 5))
        common_words.plot(kind='bar', color='green', alpha=0.7)
        plt.title('Top 20 Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.show()

    def topic_modeling(self, n_topics=10, n_top_words=10):
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(self.df['text'])
        nmf = NMF(n_components=n_topics, random_state=42).fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names_out()

        topics = self._display_topics(nmf, tf_feature_names, n_top_words)
        print("\nIdentified Topics:")
        for i, topic in enumerate(topics):
            print(f"Topic {i + 1}: {topic}")
        
        return nmf, tf

    def _display_topics(self, model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topics.append(" ".join(topic_words))
        return topics

    def detect_chapters(self, nmf_model, tf_matrix, time_threshold=60):
        topic_distribution = nmf_model.transform(tf_matrix)
        topic_distribution_trimmed = topic_distribution[:len(self.df)]
        self.df['dominant_topic'] = topic_distribution_trimmed.argmax(axis=1)

        logical_breaks = []
        for i in range(1, len(self.df)):
            if self.df['dominant_topic'].iloc[i] != self.df['dominant_topic'].iloc[i - 1]:
                logical_breaks.append(self.df['start'].iloc[i])

        consolidated_breaks = []
        last_break = None

        for break_point in logical_breaks:
            if last_break is None or break_point - last_break >= time_threshold:
                consolidated_breaks.append(break_point)
                last_break = break_point

        final_chapters = []
        last_chapter = (consolidated_breaks[0], self.df['dominant_topic'][0])

        for break_point in consolidated_breaks[1:]:
            current_topic = self.df[self.df['start'] == break_point]['dominant_topic'].values[0]
            if current_topic == last_chapter[1]:
                last_chapter = (last_chapter[0], current_topic)
            else:
                final_chapters.append(last_chapter)
                last_chapter = (break_point, current_topic)

        final_chapters.append(last_chapter)

        chapter_points = []
        chapter_names = []

        for i, (break_point, topic_idx) in enumerate(final_chapters):
            chapter_time = pd.to_datetime(break_point, unit='s').strftime('%H:%M:%S')
            chapter_points.append(chapter_time)

            chapter_text = self.df[(self.df['start'] >= break_point) & 
                                 (self.df['dominant_topic'] == topic_idx)]['text'].str.cat(sep=' ')

            vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
            tfidf_matrix = vectorizer.fit_transform([chapter_text])
            feature_names = vectorizer.get_feature_names_out()
            chapter_name = " ".join(feature_names)

            chapter_names.append(f"Chapter {i+1}: {chapter_name}")

        return list(zip(chapter_points, chapter_names))