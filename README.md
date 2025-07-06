# Kelompok Praktikum Pembelajaran Mesin - Kelas C
- 1227050027 - Azalia Fathimah Dinah
- 1227050030 - Citra Aulia
- 1227050040 - Dzilan Nazira Zahratunnisa

# SMS Spam Classification
Proyek ini merupakan implementasi berbagai model Machine Learning untuk mengklasifikasikan SMS sebagai **Spam** atau **Ham** (bukan spam). Dataset yang digunakan berasal dari Kaggle: *SMS Spam Collection - A More Diverse Dataset* dan dapat diakses melalui link berikut: https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset 

## Deskripsi
Kami membandingkan performa 3 model berbeda:
- Naive Bayes
- Decision Tree
- Logistic Regression

Evaluasi model menggunakan metrik sebagai berikut:
- Akurasi
- Precision, Recall, F1-Score
- AUC
- Confusion Matrix

## Dataset
Dataset yang digunakan: `train.csv`  
Jumlah data: 5574 SMS  
Distribusi:  
- Ham (label 0): 4518  
- Spam (label 1): 653

## Hasil Evaluasi

| Model                 |Akurasi | Precision | Recall | F1-Score | AUC   |
|-----------------------|--------|-----------|--------|----------|-------|
| Decision Tree         | 95.46% | 0.85      | 0.77   | 0.81     | 0.87  |
| Naive Bayes           | 96.32% | 1         | 0.72   | 0.83     | 0.97  |
| Logistic Regression   | 97.04% | 0.98      | 0.81   | 0.89     | 0.91  |

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- seaborn
- matplotlib

## Link Notebook & Hasil Model
- decission tree: https://www.kaggle.com/code/citrlia/decissiontree 
- naive bayes: https://www.kaggle.com/code/dzilann/naive-bayes-spam 
- logistic regression: https://colab.research.google.com/drive/1QU7RxH4mrxpxw85-B353QK5zAspEovnR?usp=sharing
