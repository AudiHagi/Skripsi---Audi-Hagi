import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

df = pd.read_csv("Processed Data/Data Preprocessing.csv", delimiter=';')
df = df.dropna(subset=['label'])
print("Data Shape Before: ", df.shape)
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
negative_percentage = int(round(total_negatif / len(df) * 100,0))
positive_percentage = int(round(total_positif / len(df) * 100,0))
print(f"Negative Sentiments Before: {negative_percentage}% / {total_negatif}")
print(f"Positive Sentiments Before: {positive_percentage}% / {total_positif}")
df.to_csv("Processed Data/Data Modelling Before Z-Score.csv", index=False, sep=";")
print("----- data normalization start -----")
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
negatif_zscore = zscore([total_negatif, total_positif])[0]
positif_zscore = zscore([total_negatif, total_positif])[1]
threshold_zscore = 0.005
if negatif_zscore < threshold_zscore:
  diff_negatif = int((threshold_zscore - negatif_zscore) * len(df))
  additional_negatif_samples = df[df['label'] == 0].sample(n=diff_negatif, replace=True)
  df = pd.concat([df, additional_negatif_samples], ignore_index=True)
if positif_zscore < threshold_zscore:
  diff_positif = int((threshold_zscore - positif_zscore) * len(df))
  additional_positif_samples = df[df['label'] == 1].sample(n=diff_positif, replace=True)
  df = pd.concat([df, additional_positif_samples], ignore_index=True)
print("Data Shape After: ", df.shape)
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
negative_percentage = int(round(total_negatif / len(df) * 100,0))
positive_percentage = int(round(total_positif / len(df) * 100,0))
print(f"Negative Sentiments After: {negative_percentage}% / {total_negatif}")
print(f"Positive Sentiments After: {positive_percentage}% / {total_positif}")
print("----- data normalization finish -----")
df.to_csv("Processed Data/Data Modelling After Z-Score.csv", index=False, sep=";")

print("----- sentiment count visualization start -----")
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
labels = ['Negatif', 'Positif']
counts = [total_negatif, total_positif]
colors = ['#f72323', '#2394f7']
sns.set_theme()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(counts, labels=labels, autopct=lambda pct: f"{int(round(pct,0))}%", startangle=90, colors=colors)
ax.axis('equal')
ax.set_title('Visualisasi Sentimen Negatif dan Positif Setelah Z-Score')
fig.savefig('Visualization/Sentiment Count Chart After Z-Score.png')
plt.savefig("Website/static/assets/images/Sentiment Count Chart After Z-Score.png")
print("----- sentiment count visualization finish -----")