import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Membuat dataset
data = {
    'GPA': [3.5, 3.0, 3.8, 2.8, 3.7, 3.2],
    'Skor_Tes': [85, 80, 90, 75, 88, 78],
    'Rekomendasi': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak'],
    'Diterima': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak']
}

df = pd.DataFrame(data)

# Encode data kategorikal
df['Rekomendasi'] = df['Rekomendasi'].map({'Ya': 1, 'Tidak': 0})
df['Diterima'] = df['Diterima'].map({'Ya': 1, 'Tidak': 0})

df

class NaiveBayesClassifier:
    def __init__(self):
        self.model = GaussianNB()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def accuracy(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)
    
# Membagi data menjadi fitur dan target
X = df[['GPA', 'Skor_Tes', 'Rekomendasi']]
y = df['Diterima']

# Membagi data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Membuat instance dari NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier()

# Melatih model
nb_classifier.train(X_train, y_train)

# Membuat prediksi
y_pred = nb_classifier.predict(X_test)

# Menghitung akurasi
accuracy = nb_classifier.accuracy(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

# Menampilkan prediksi
predictions = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
print(predictions)

# Membuat prediksi untuk data baru
data_baru = pd.DataFrame({'GPA': [3.6], 'Skor_Tes': [82], 'Rekomendasi': [1]})
prediksi_baru = nb_classifier.predict(data_baru)
print(f'Prediksi untuk data baru (GPA=3.6, Skor Tes=82, Rekomendasi=Ya): {"Diterima" if prediksi_baru[0] == 1 else "Tidak Diterima"}')