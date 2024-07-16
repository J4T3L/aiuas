import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

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

class ANNClassifier:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=100, batch_size=10):
        # Menggunakan callback untuk mencetak loss dan akurasi setiap epoch
        class PrintEpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"Epoch {epoch + 1}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[PrintEpochCallback()])

    def predict(self, X_test):
        return (self.model.predict(X_test) > 0.5).astype("int32")

    def accuracy(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)

# Membagi data menjadi fitur dan target
X = df[['GPA', 'Skor_Tes', 'Rekomendasi']]
y = df['Diterima']

# Membagi data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat instance dari ANNClassifier
ann_classifier = ANNClassifier(input_dim=X_train.shape[1])

# Melatih model dengan 100 epoch
ann_classifier.train(X_train, y_train, epochs=100)

# Membuat prediksi
y_pred = ann_classifier.predict(X_test)

# Menghitung akurasi
accuracy = ann_classifier.accuracy(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

# Menampilkan prediksi
predictions = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred.flatten()})
print(predictions)

# Membuat prediksi untuk data baru
data_baru = pd.DataFrame({'GPA': [3.6], 'Skor_Tes': [82], 'Rekomendasi': [1]})
data_baru = scaler.transform(data_baru)
prediksi_baru = ann_classifier.predict(data_baru)
print(f'Prediksi untuk data baru (GPA=3.6, Skor Tes=82, Rekomendasi=Ya): {"Diterima" if prediksi_baru[0][0] == 1 else "Tidak Diterima"}')
