import joblib
import swifter
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from google_play_scraper import reviews_all, Sort
from flask import flash, render_template, send_file, redirect, request, Flask
import sys
sys.path.append('../')
import preprocessing

app = Flask(__name__)
app.secret_key = '@#$123456&*()'
downloading = False

def preprocess_file_predict(review):
  cleaned_comment = preprocessing.cleaning_comment(review)
  tokenized_comment = preprocessing.tokenize_comment(cleaned_comment)
  stemmed_comment = preprocessing.stemming_comment(tokenized_comment, "../Kamus/Skip-Stemming-Words-Dict.txt", "../Kamus/Skip-Elongation-Words-Dict.txt")
  combined_comment = preprocessing.combining_words(stemmed_comment, "../Kamus/Combine-Words-Dict.txt")
  removed_stopword_comment = preprocessing.remove_stopword_comment(combined_comment, "../Kamus/Skip-Stopwords-Dict.txt", "../Kamus/Stopwords-Bahasa-Dict.txt")
  return removed_stopword_comment

def crawl_reviews(count, from_date, to_date):
  while True:
    sirekap = reviews_all(
      "id.go.kpu.sirekap2024",
      lang="id",
      country="",
      sort=Sort.MOST_RELEVANT, 
      filter_score_with=None
    )
    sirekap_df = pd.DataFrame(sirekap)
    sirekap_df.to_csv("Static/assets/temp/SiRekap Google Play Review.csv", index=None, header=True)
    df = pd.read_csv("Static/assets/temp/SiRekap Google Play Review.csv")
    df = df.rename(columns={"content": "review", "at": "date"})
    start_date = f"{from_date} 00:00:01"
    end_date = f"{to_date} 23:59:59"
    filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = filtered_df[["review", "date"]]
    df.to_csv("Static/assets/temp/SiRekap Google Play Review Custom.csv", index=None, header=True)
    df = pd.read_csv("Static/assets/temp/SiRekap Google Play Review Custom.csv")
    df = df[["review"]]
    if len(df) >= int(count) or not downloading:
      break
  return df

feature_pkl = open('static/assets/pickles/Vector LR.pkl', 'rb')
model_ori_pkl = open('static/assets/pickles/Model LR.pkl', 'rb')

feature = joblib.load(feature_pkl)
model_ori = joblib.load(model_ori_pkl)

@app.route('/')
def home():
  global downloading
  downloading = False
  with open('static/assets/documents/classification_report.txt', 'r') as file:
    lines = file.readlines()
  metrics_dict = {}
  for line in lines:
    key, value = line.strip().split(': ')
    metrics_dict[key] = value
  model_information_dict = {
    'negative_percentage': metrics_dict['negative_percentage'],
    'positive_percentage': metrics_dict['positive_percentage'],
    'total_negatif': metrics_dict['total_negatif'],
    'total_positif': metrics_dict['total_positif'],
    'accuracy': metrics_dict['accuracy'],
    'precision_negative': metrics_dict['precision_negative'],
    'precision_positive': metrics_dict['precision_positive'],
    'recall_negative': metrics_dict['recall_negative'],
    'recall_positive': metrics_dict['recall_positive'],
    'f1_negative': metrics_dict['f1_negative'],
    'f1_positive': metrics_dict['f1_positive'],
    }
  return render_template('home.html', home_active=True, fe_model_information_dict=model_information_dict)

@app.route('/dataTrain')
def dataTrain():
  global downloading
  downloading = False
  df = pd.read_csv("static/assets/documents/Data Train.csv", delimiter=';')
  df['label'] = df['label'].map({
    0: "<p class='px-2 font-bold text-center' style='color: red;'>Negatif</p>",
    1: "<p class='px-2 font-bold text-center' style='color: blue;'>Positif</p>"
    })
  html_content = df.to_html(index=True, escape=False)
  return render_template('data_train.html', fe_html_content=html_content, data_train_active=True)

@app.route('/download')
def download():
  global downloading
  downloading = False
  return render_template('download_dataset.html', download_active=True)

@app.route('/downloaddataset', methods=['POST', 'GET'])
def downloaddataset():
  try:
    if request.method == 'POST':
      count = request.form['review_count']
      from_date = request.form['date_from_input']
      to_date = request.form['date_to_input']
      global downloading
      downloading = True
      df = crawl_reviews(count, from_date, to_date)
      df = df.iloc[:int(count)]
      df['label'] = ''
      filtered_csv_path = "Static/assets/temp/SiRekap user App Comment.csv"
      df.to_csv(filtered_csv_path, sep=';', index=None, header=True)
      file_name = f"SiRekap user App Comment {count}.csv"
      return send_file(filtered_csv_path, as_attachment=True, download_name=file_name)
  except Exception as e:
      flash(f'Unduh Ulasan Aplikasi SiRekap Gagal ! Error: {str(e)}', 'error')
      return redirect('/download') 

@app.route('/singlepredict')
def singlepredict():
  global downloading
  downloading = False
  return render_template('single_predict.html', single_predict_active=True)

@app.route('/singlepredictresult', methods=['POST', 'GET'])
def singlepredictresult():
  global downloading
  downloading = False
  try:
    if request.method == 'POST':
      review = request.form['review']
      cleaned_comment = preprocessing.cleaning_comment(review)
      tokenized_comment = preprocessing.tokenize_comment(cleaned_comment)
      stemmed_comment = preprocessing.stemming_comment(tokenized_comment, "../Kamus/Skip-Stemming-Words-Dict.txt", "../Kamus/Skip-Elongation-Words-Dict.txt")
      combined_comment = preprocessing.combining_words(stemmed_comment, "../Kamus/Combine-Words-Dict.txt")
      removed_stopword_comment = preprocessing.remove_stopword_comment(combined_comment, "../Kamus/Skip-Stopwords-Dict.txt", "../Kamus/Stopwords-Bahasa-Dict.txt")
      data = [removed_stopword_comment]
      vector = feature.transform(data)
      review_prediction = model_ori.predict(vector)[0]
      flash('Prediksi Ulasan Tunggal Aplikasi SiRekap Berhasil !', "success")
      return render_template('single_predict_result.html', single_predict_active=True, fe_review=review, fe_prediction=review_prediction, fe_cleaned_comment=cleaned_comment, fe_combined_comment=combined_comment, fe_tokenized_comment=tokenized_comment, fe_stemmed_comment=stemmed_comment, fe_removed_stopword_comment=removed_stopword_comment)
  except Exception as e:
    flash(f'Prediksi Ulasan Tunggal Aplikasi SiRekap Gagal ! Error: {str(e)}', "error")
    return redirect('/singlepredict')

@app.route('/filepredict')
def filepredict():
  global downloading
  downloading = False
  return render_template('file_predict.html', bulk_predict_active=True)

@app.route('/filepredictresult', methods=['POST', 'GET'])
def filepredictresult():
  global downloading
  downloading = False
  if request.method == 'POST':
    csv_file = request.files.get("file")
    df = pd.read_csv(csv_file, delimiter=';')
    print("Jumlah Data ", len(df))
    if len(df) >= 50:
      try:
        start_time = time.time()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        original_sentiment = df['label']
        original_comment = df['review']
        df['review'] = df['review'].swifter.apply(preprocess_file_predict)
        data = df['review'].tolist()
        vectors = feature.transform(data)
        predictions = model_ori.predict(vectors)
        sentiments = []
        sentiments_html = []
        for pred in predictions:
          if pred == 0:
            sentiments.append("Negative")
            sentiments_html.append("<p class='px-2 font-bold text-center' style='color: red;'>Negatif</p>")
          else:
            sentiments.append("Positive")
            sentiments_html.append("<p class='px-2 font-bold text-center' style='color: blue;'>Positif</p>")
        metrics_dict = {
          'precision_negative': int(precision_score(original_sentiment, predictions, labels=[0])* 100),
          'recall_negative': int(recall_score(original_sentiment, predictions, labels=[0])* 100),
          'f1_negative': int(f1_score(original_sentiment, predictions, labels=[0])* 100),
          'precision_positive': int(precision_score(original_sentiment, predictions, labels=[1])* 100),
          'recall_positive': int(recall_score(original_sentiment, predictions, labels=[1])* 100),
          'f1_positive': int(f1_score(original_sentiment, predictions, labels=[1])* 100),
          'accuracy': int(accuracy_score(original_sentiment, predictions)* 100),
          }
        df['label'] = df['label'].map({0: 'Negative', 1: 'Positive'})
        df['review'] = original_comment
        df['sentimen'] = sentiments
        csv_data = df.to_csv(sep=';', index=None, header=True)
        df['label'] = df['label'].map({
          'Negative': "<p class='px-2 font-bold text-center' style='color: red;'>Negatif</p>",
          'Positive': "<p class='px-2 font-bold text-center' style='color: blue;'>Positif</p>"
          })
        df['sentimen'] = sentiments_html
        html_content = df.to_html(index=True, escape=False)
        print("----- visualization confussion matrix heatmap start -----")
        cm = confusion_matrix(original_sentiment, predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        sns.set_theme()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix Heatmap Prediction")
        plt.savefig("static/assets/images/Confusion Matrix Heatmap Prediction.png")
        print("----- visualization confussion matrix heatmap finish -----")
        end_time = time.time()
        duration_second = end_time - start_time
        duration_total = time.strftime("%H:%M:%S", time.gmtime(duration_second))
        flash('Prediksi File Ulasan Aplikasi SiRekap Berhasil !')
        return render_template('file_predict_result.html', bulk_predict_active=True, fe_html_content=html_content, fe_csv_data=csv_data, fe_duration_total=duration_total, fe_metrics_dict=metrics_dict)
      except Exception as e:
        flash(f'Prediksi File Ulasan Aplikasi SiRekap Gagal ! Error: {str(e)}', "error")
        return redirect('/filepredict')
    else:
      flash('Prediksi File Ulasan Aplikasi SiRekap Gagal ! Jumlah Data Dalam File Kurang Dari (<) 50', "error")
      return redirect('/filepredict')

if __name__ == '__main__':
  app.run(debug=True)