import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize

def negative_wordcloud_and_bar_chart(data):
  output_path_wc = "Visualization/Wordcloud Kata Negatif.png"
  output_path_wc_web = "Website/static/assets/images/Wordcloud Kata Negatif.png"
  output_path_bar_chart = "Visualization/Bar Chart Kata Negatif.png"
  data_negatif = data[data['label'] == 0]
  text_negatif = ' '.join(str(word) for word in data_negatif['review'].fillna(""))
  wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(text_negatif)
  wordcloud.to_file(output_path_wc)
  wordcloud.to_file(output_path_wc_web)
  words = word_tokenize(text_negatif)
  word_counts = Counter(words)
  top_words = word_counts.most_common(20)
  plt.figure(figsize=(10, 6))
  words, counts = zip(*top_words)
  plt.barh(words, counts, color='#f72323')
  plt.xlabel('Frequency')
  plt.ylabel('Words')
  plt.title('Top 20 Most Frequent Negative Words')
  plt.savefig(output_path_bar_chart)

def positive_wordcloud_and_bar_chart(data):
  output_path_wc = "Visualization/Wordcloud Kata Positif.png"
  output_path_wc_web = "Website/static/assets/images/Wordcloud Kata Positif.png"
  output_path_bar_chart = "Visualization/Bar Chart Kata Positif.png"
  data_positif = data[data['label'] == 1]
  text_positif = ' '.join(str(word) for word in data_positif['review'].fillna(""))
  wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(text_positif)
  wordcloud.to_file(output_path_wc)
  wordcloud.to_file(output_path_wc_web)
  words = word_tokenize(text_positif)
  word_counts = Counter(words)
  top_words = word_counts.most_common(20)
  plt.figure(figsize=(10, 6))
  words, counts = zip(*top_words)
  plt.barh(words, counts, color='#2394f7')
  plt.xlabel('Frequency')
  plt.ylabel('Words')
  plt.title('Top 20 Most Frequent Positive Words')
  plt.savefig(output_path_bar_chart)

def sentiment_countplot(data):
  output_path="Visualization/Sentiment Count Chart Before Z-Score.png"
  total_negatif = data['label'].value_counts()[0]
  total_positif = data['label'].value_counts()[1]
  labels = ['Negatif', 'Positif']
  counts = [total_negatif, total_positif]
  colors = ['#f72323', '#2394f7']
  sns.set_theme()
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.pie(counts, labels=labels, autopct=lambda pct: f"{int(round(pct,0))}%", startangle=90, colors=colors)
  ax.axis('equal')
  ax.set_title('Visualisasi Sentimen Negatif dan Positif Sebelum Z-Score')
  fig.savefig(output_path)

df = pd.read_csv("Processed Data/Data Preprocessing.csv", delimiter=';')
print("----- data visualization start -----")
negative_wordcloud_and_bar_chart(df)
positive_wordcloud_and_bar_chart(df)
sentiment_countplot(df)
print("----- data visualization finish -----")