# Study Case Machine Learning

## 1.	Data Wrangling & EDA
Tahapan ini merupakan tahap awal yang dilakukan sebelum adanya proses transformasi pada data. kedua proses ini bertujuan untuk mengenali **karakteristik data** lebih lanjut. 

**data wrangling** dapat didefinisikan sebagai proses untuk mengubah data mentah menjadi bentuk atau format yang bisa diproses pada tahap selanjutnya. Contoh penerapannya adalah sebagai berikut

```python
dataset = pd.read_csv('/content/drive/MyDrive/python/titanic.csv')
dataset.head()
```

sedangkan **EDA** atau *exploratory data analysis* merupakan proses analisis untuk memahami karakteristik data untuk kemudian dapat dijadikan acuan untuk menyiapkan data sebelum akhirnya masuk proses modelling. Contohnya adalah sebagai berikut.

```python
dataplot=sns.heatmap(dataset.corr(), annot=True)
plt.show()
```

## 2.	Data Splitting
Apabila diartikan secara harfiah *data splitting* memiliki arti membagi data. Hal tersebut sudah mewakili proses yang terjadi pada tahapan ini yaitu proses pembagian data menjadi subhimpunan data, dan umumnya data akan dibagi menjadi data latih dan data uji untuk tahap evaluasi model. Penerapannya adalah sebagai berikut

```python
X = dataset.drop(['Survived'], axis=1)
y = dataset['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3.	Data Preprocessing
Data Preprocessing merupakan salah satu tahapan penting untuk mempersiapkan data sebelum data tersebut masuk pada proses selanjutnya. Data yang merupakan data mentah akan diolah terlebih dahulu. Umumnya data mentah memiliki karakteristik yang tidak konsisten dan memiliki nilai yang hilang. Preprocessing ini sangat penting untuk menangani nilai yang hilang dan mengatasi inkonsistensi. Data Preprocessing atau praproses data dapat dilakukan dengan cara eliminasi terkait data yang tidak sesuai. Contoh penghapusan baris dengan nilai kosong adalah sebagai berikut

```python
dataset = dataset.dropna()
```

## 4.	Feature Engineering
Feature engineering merupakan salah satu tahapan yang cukup penting dalam proses pengaplikasian machine learning. Sebagian besar performa machine learning sangat bergantung pada representasi dari feature pada data, dan umumnya model tidak dapat memproses data dengan tipe data *object*. Salah satu teknik yang dapat dilakukan untuk mengekstraksi fitur pada data adalah dengan menerapkan One-Hot Encoder yang akan membuat kolom baru berdasarkan kelas pada kolom kategorikal. Contoh penerapan feature engineering One-Hot Encoder adalah sebagai berikut.

```python
dataset = pd.get_dummies(dataset, columns=['Sex'])
```

## 5.	Model Training
Proses ini merupakan tahapan untuk melatih model dengan menggunakan data latih yang telah ditetapkan sebelumnya berdasarkan dataset. Proses ini dapat diulangi kembali apabila hasil evaluasi menunjukkan hasil yang kurang baik. Berikut adalah contoh penerapan algoritma Random Forest Classifier 

```python
clf = RandomForestClassifier(n_estimators = 200, max_depth = 4,
                             random_state = 18).fit(X_train, y_train)
```

## 6.	Model Evaluation
merupakan proses evaluasi model yang dilakukan untuk menemukan model terbaik yang mewakili dataset atau studi kasus dan sekaligus mengetahui seberapa baik model tersebut bekerja. Evaluasi yang umum digunakan pada klasifikasi adalah dengan confusion matrix atau classification report. Berikut adalah contoh penerapan dari classification report

```python
print(classification_report(y_test,predictions))
```
