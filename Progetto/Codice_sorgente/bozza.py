import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Funzioni per caricare i dati
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            file_name = parts[0]
            label = parts[1]
            data.append((file_name, label))
    return data

def load_evaluation_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            file_name = parts[0]
            label = parts[1]
            start_time = float(parts[2])
            end_time = float(parts[3])
            event = parts[4]
            data.append((file_name, label, start_time, end_time, event))
    return data

def extract_features(file_name):
    y, sr = librosa.load(file_name)
    features = librosa.feature.mfcc(y=y, sr=sr)
    return features.flatten()

def extract_segment_features(file_name, start_time, end_time):
    y, sr = librosa.load(file_name, sr=None)
    y_segment = y[int(start_time * sr):int(end_time * sr)]
    features = librosa.feature.mfcc(y=y_segment, sr=sr)
    return features.flatten()

# Carica i dati di addestramento e test
train_data = load_data('/mnt/data/street_fold3_train.txt')
test_data = load_data('/mnt/data/street_fold3_test.txt')
evaluate_data = load_evaluation_data('/mnt/data/street_fold4_evaluate.txt')

# Estrai le caratteristiche per il training
X_train = [extract_features(file_name) for file_name, label in train_data]
y_train = [label for file_name, label in train_data]

# Addestra il modello
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Estrai le caratteristiche per il testing
X_test = [extract_features(file_name) for file_name, label in test_data]
y_test = [label for file_name, label in test_data]

# Valuta il modello sui dati di test
predictions_test = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions_test)
print(f"Test Accuracy: {accuracy}")

# Estrai le caratteristiche per l'evaluation e fai predizioni
X_eval = [extract_segment_features(file_name, start_time, end_time) for file_name, label, start_time, end_time, event in evaluate_data]
y_eval = [event for file_name, label, start_time, end_time, event in evaluate_data]

# Fai predizioni per l'evaluation
predictions_eval = model.predict(X_eval)

# Calcola le metriche di valutazione dettagliate
print(classification_report(y_eval, predictions_eval))
