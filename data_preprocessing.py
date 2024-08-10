import pandas as pd
import swifter
import preprocessing

def pipeline_preprocessing(df, column_name):
  df[column_name] = df[column_name].swifter.apply(preprocessing.cleaning_comment)
  df = preprocessing.preprocess_dataframe(df, column_name)
  df.to_csv("Processed Data/Data Cleaning.csv", index=False, sep=";")
  df[column_name] = df[column_name].swifter.apply(preprocessing.tokenize_comment)
  df = preprocessing.preprocess_dataframe(df, column_name)
  df.to_csv("Processed Data/Data Tokenize.csv", index=False, sep=";")
  df[column_name] = df[column_name].swifter.apply(lambda x: preprocessing.stemming_comment(x, "Kamus/Skip-Stemming-Words-Dict.txt", "Kamus/Skip-Elongation-Words-Dict.txt"))
  df = preprocessing.preprocess_dataframe(df, column_name)
  df.to_csv("Processed Data/Data Stemming.csv", index=False, sep=";")
  df[column_name] = df[column_name].swifter.apply(lambda x: preprocessing.combining_words(x,"Kamus/Combine-Words-Dict.txt",))
  df = preprocessing.preprocess_dataframe(df, column_name)
  df.to_csv("Processed Data/Data Word Combining.csv", index=False, sep=";")
  df[column_name] = df[column_name].swifter.apply(lambda x: preprocessing.remove_stopword_comment(x,"Kamus/Skip-Stopwords-Dict.txt", "Kamus/Stopwords-Bahasa-Dict.txt"))
  df = preprocessing.preprocess_dataframe(df, column_name)
  df.to_csv("Processed Data/Data Stopword.csv", index=False, sep=";")
  return df

df = pd.read_csv("Data Set/Data Train/Sirekap User App Comment.csv", delimiter=';')
print("Data Shape Before: ", df.shape)
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
negative_percentage = int(round(total_negatif / len(df) * 100,0))
positive_percentage = int(round(total_positif / len(df) * 100,0))
print(f"Negative Sentiments Before: {negative_percentage}% / {total_negatif}")
print(f"Positive Sentiments Before: {positive_percentage}% / {total_positif}")
df = pipeline_preprocessing(df, 'review')
print("Data Shape After: ", df.shape)
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
negative_percentage = int(round(total_negatif / len(df) * 100,0))
positive_percentage = int(round(total_positif / len(df) * 100,0))
print(f"Negative Sentiments After: {negative_percentage}% / {total_negatif}")
print(f"Positive Sentiments After: {positive_percentage}% / {total_positif}")
df.to_csv("Processed Data/Data Preprocessing.csv", index=False, sep=";")
df.to_csv("Website/static/assets/documents/Data Train.csv", index=False, sep=";")