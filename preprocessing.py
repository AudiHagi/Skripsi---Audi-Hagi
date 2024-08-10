import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_word_elongation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_dataframe(df, column_name):
  df.dropna(subset=[column_name], inplace=True)
  df.drop_duplicates(subset=[column_name], inplace=True)
  return df

def read_words_from_file(file_path):
  with open(file_path, 'r') as file:
    words = file.read().splitlines()
  return words

def cleaning_comment(text):
  # Mengubah teks menjadi lowercase
  lowercase_text = text.lower()
  # Menghilangkan new line
  no_newline_text = lowercase_text.replace('\n', ' ')
  # Menghilangkan karakter yang bukan angka atau huruf
  alphanumeric_text = re.sub(r'[^a-zA-Z0-9]', ' ', no_newline_text)
  # Menghilangkan angka
  no_number_text = re.sub(r"\d+", " ", alphanumeric_text)
  # Menghilangkan extra whitespace
  single_whitespace_text = re.sub(r'\s\s+', ' ', no_number_text)
  # Menghilangkan whitespace di awal dan akhir teks
  cleaned_text = single_whitespace_text.strip()
  return cleaned_text

def tokenize_comment(text):
  # Mengubah text menjadi token ['']
  tokenized_text = word_tokenize(text)
  return tokenized_text

def stemming_comment(text, skip_words_stem_file, skip_elongation_file):
  skip_words_stem = read_words_from_file(skip_words_stem_file)
  skip_elongation = read_words_from_file(skip_elongation_file)
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  stemmed_words = []
  for word in text:
    if word not in skip_words_stem:
      # Stemming kata
      stemmed_word = stemmer.stem(word)
      if stemmed_word not in skip_elongation:
        # Replace elongation word
        stemmed_word = replace_word_elongation(stemmed_word)
    else:
      stemmed_word = word
    stemmed_words.append(stemmed_word)
  stemmed_text = []
  stemmed_text = " ". join(stemmed_words)
  return stemmed_text

def combining_words(text, combine_words_file):
  normalisasi_dict = {}
  with open(combine_words_file, "r") as NORMALIZE_WORD:
    for line in NORMALIZE_WORD:
      key, value = line.strip().split(":", 1)
      normalisasi_dict[key] = value
  for key, value in normalisasi_dict.items():
    text = re.sub(r'\b{}\b'.format(key), value, text)
  text = re.sub(r'\s\s+', ' ', text)
  text = text.strip()
  return text

def remove_stopword_comment(text, skip_stopword_file, stopwords_file):
  skip_stopword = read_words_from_file(skip_stopword_file)
  list_stopwords = stopwords.words('indonesian')
  with open(stopwords_file, "r") as STOPWORDS:
    txt_stopwords = STOPWORDS.read().splitlines()
    list_stopwords.extend(txt_stopwords)
  words = text.split() 
  filtered_words = []
  for word in words:
    if word not in list_stopwords or word in skip_stopword:
      filtered_words.append(word)
  removed_stopword_text = ' '.join(filtered_words).strip()
  return removed_stopword_text