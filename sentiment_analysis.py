import subprocess
import time

print("----- SENTIMENT ANALYSIS START -----\n")
start_time = time.time()

print("----- DATA SCRAPPING START -----")
subprocess.call(["python", "data_scrapping.py"])
print("----- DATA SCRAPPING FINISH -----\n")

print("----- DATA PREPROCESSING START -----")
subprocess.call(["python", "data_preprocessing.py"])
print("----- DATA PREPROCESSING FINISH -----\n")

print("----- DATA NORMALIZATION START -----")
subprocess.call(["python", "data_normalization.py"])
print("----- DATA NORMALIZATION FINISH -----\n")

print("----- DATA MODELLING START -----")
subprocess.call(["python", "data_modelling.py"])
print("----- DATA MODELLING FINISH -----\n")

print("----- DATA VISUALIZATION START -----")
subprocess.call(["python", "data_visualization.py"])
print("----- DATA VISUALIZATION FINISH -----\n")

end_time = time.time()
duration_second = end_time - start_time
duration_total = time.strftime("%H:%M:%S", time.gmtime(duration_second))
print("----- SENTIMENT ANALYSIS FINISH -----")
print("Durasi Total Proses (m) : ", duration_total)