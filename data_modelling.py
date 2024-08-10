import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Processed Data/Data Modelling After Z-Score.csv", delimiter=';')
print("----- split data start -----")
comment_text = df['review']
sentiment_label = df['label']
x_train, x_test, y_train, y_test = train_test_split(comment_text, sentiment_label, test_size=0.2, stratify=sentiment_label, random_state=42)
x_train.fillna('', inplace=True)
x_test.fillna('', inplace=True)
print("----- split data finish -----")

print("----- tf-idf start -----")
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectorizer = tf_idf_vectorizer.fit(x_train)
x_train_tfidf = tf_idf_vectorizer.fit_transform(x_train)
x_test_tfidf = tf_idf_vectorizer.transform(x_test)
data_tf_idf = pd.DataFrame(x_test_tfidf.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
with open('Website/static/assets/pickles/Vector LR.pkl', 'wb') as output:
  pickle.dump(tf_idf_vectorizer, output)
print("----- tf-idf finish -----")

print("----- data modelling logistic regression start -----")
logistic_regression = LogisticRegression()
lr_model = logistic_regression.fit(x_train_tfidf, y_train)
with open('Website/static/assets/pickles/Model LR.pkl', 'wb') as output:
  pickle.dump(lr_model, output)
model_pred = lr_model.predict(x_test_tfidf)
prediksi_benar = (model_pred == y_test).sum()
print("Jumlah Prediksi Benar: ", prediksi_benar)
prediksi_salah = (model_pred != y_test).sum()
print("Jumlah Prediksi Salah: ", prediksi_salah)
con_mat = confusion_matrix(y_test, model_pred)
print("Confusion Matrix:\n", con_mat)
classification_rep = classification_report(y_test, model_pred, zero_division=1)
print("Laporan Klasifikasi:\n", classification_rep)
acc_score = int(round(accuracy_score(model_pred, y_test) * 100,0))
print('Accuracy: ', acc_score)
total_negatif = df['label'].value_counts()[0]
total_positif = df['label'].value_counts()[1]
negative_percentage = int(round(total_negatif / len(df) * 100,0))
positive_percentage = int(round(total_positif / len(df) * 100,0))
precision_negative =  int(round(precision_score(y_test, model_pred, pos_label=0)* 100))
precision_positive =  int(round(precision_score(y_test, model_pred, pos_label=1)* 100))
recall_negative =  int(round(recall_score(y_test, model_pred, pos_label=0)* 100))
recall_positive =  int(round(recall_score(y_test, model_pred, pos_label=1)* 100))
f1_negative =  int(round(f1_score(y_test, model_pred, pos_label=0)* 100))
f1_positive =  int(round(f1_score(y_test, model_pred, pos_label=1)* 100))
report_dict = {}
report_dict["negative_percentage"] = negative_percentage
report_dict["positive_percentage"] = positive_percentage
report_dict["total_negatif"] = total_negatif
report_dict["total_positif"] = total_positif
report_dict["accuracy"] = acc_score
report_dict["precision_negative"] = precision_negative
report_dict["precision_positive"] = precision_positive
report_dict["recall_negative"] = recall_negative
report_dict["recall_positive"] = recall_positive
report_dict["f1_negative"] = f1_negative
report_dict["f1_positive"] = f1_positive
with open("Website/static/assets/documents/classification_report.txt", "w") as file:
  for key, value in report_dict.items():
    file.write(f"{key}: {value}\n")
print("----- data modelling logistic regression finish -----")

print("----- logistic regression visualization confussion matrix heatmap start -----")
plt.figure(figsize=(10, 10))
sns.heatmap(con_mat, annot=True, fmt="d", cmap="Blues")
sns.set_theme()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap LR")
plt.savefig("Visualization/Confusion Matrix Heatmap LR.png")
plt.savefig("Website/static/assets/images/Confusion Matrix Heatmap LR.png")
print("----- logistic regression visualization confussion matrix heatmap finish -----")