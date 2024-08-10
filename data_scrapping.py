import pandas as pd
from google_play_scraper import Sort, reviews_all

print("----- data review scrapping start -----")
sirekap= reviews_all("id.go.kpu.sirekap2024", lang="id", sort=Sort.MOST_RELEVANT, filter_score_with=None)
sirekap = pd.DataFrame(sirekap)
sirekap.to_csv("Data Set/SiRekap Google Play Review.csv", index=None, header=True)
df = pd.read_csv("Data Set/SiRekap Google Play Review.csv")
print("Data Shape : ", df.shape)
df = df.rename(columns={"content": "review", "at": "date", "score":"label"})
df['label'] = df['label'].replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
df.to_csv("Data Set/SiRekap Google Play Review1.csv", index=None, header=True)
print("----- data review scrapping finish -----")

"""
Collecting Data Train
"""
print("----- data train collecting finish -----")
df = pd.read_csv("Data Set/SiRekap Google Play Review.csv")
end_date = "2024-02-13 23:59:59"
filtered_df = df[df["date"] <= end_date]
df = filtered_df[["review", "date", "label"]]
df.to_csv("Data Set/Data Train/SiRekap Google Play Review Custom.csv", sep=';', index=None, header=True)
df = pd.read_csv("Data Set/Data Train/SiRekap Google Play Review Custom.csv", sep=';')
print("Data Shape Train: ", df.shape)
df = df[["review", "label"]]
df.to_csv("Data Set/Data Train/Sirekap User App Comment.csv", sep=';', index=None, header=True)
print("----- data train collecting finish -----")

"""
Collecting Data Test
"""
print("----- data test collecting finish -----")
df = pd.read_csv("Data Set/SiRekap Google Play Review.csv")
start_date = "2024-02-14 00:00:01"
filtered_df = df[df["date"] >= start_date]
df = filtered_df[["review", "date", "label"]]
df.to_csv("Data Set/Data Test/SiRekap Google Play Review Custom Test.csv", sep=';', index=None, header=True)
df = pd.read_csv("Data Set/Data Test/SiRekap Google Play Review Custom Test.csv", delimiter=';')
print("Data Shape Test: ", df.shape)
df = df[["review", "label"]]
df.to_csv("Data Set/Data Test/Sirekap User App Comment Test.csv", sep=';', index=None, header=True)
print("----- data test collecting finish -----")