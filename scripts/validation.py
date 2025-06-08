import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = tf.keras.callbacks.EarlyStopping
import sys, os
#sys.path.append('./neutral_networks.py')  # lub ścieżka do neutral_networks.py
scripts_dir = os.path.abspath(os.path.join('neutral_networks.py', os.pardir))
sys.path.insert(0, scripts_dir)


import neutral_networks as nn

# 1) Wczytaj i połącz zbiór (np. train+val) lub użyj dataset.npy:
train = np.load('./data/datasets/standard_split/color/128x128/train_set.npy', allow_pickle=True)
val   = np.load('./data/datasets/standard_split/color/128x128/val_set.npy', allow_pickle=True)
data_all = np.concatenate([train, val], axis=0)

# 2) Rozpakuj X, y
X, y = nn.unpacking_data(data_all)

# 3) Przygotuj model CNN i callback EarlyStopping
n_classes = y.shape[1]
input_shape = X.shape[1:]  # (H, W, C)
model_cnn = nn.create_model_cnn(
    n_classes=n_classes,
    input_shape=input_shape,
    optimizer='adam',
    func_activation='relu',
    kernel_initializer='he_uniform'
)
callbacks = [EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]

#4) 5-krotna walidacja
scores_cnn, y_true_cnn, y_pred_cnn = nn.cross_validation(
    n_splits=5,
    X=X, y=y,
    model=model_cnn,
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)
print("CNN fold scores (accuracy):", scores_cnn, "mean:", np.mean(scores_cnn))

# Metryki dla CNN
for i, (yt, yp) in enumerate(zip(y_true_cnn, y_pred_cnn), 1):
    prec = precision_score(yt, yp, average='weighted', zero_division=0)
    rec = recall_score(yt, yp, average='weighted', zero_division=0)
    f1 = f1_score(yt, yp, average='weighted', zero_division=0)
    cm = confusion_matrix(yt, yp)
    print(f"Fold {i} - precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")
    print(f"Fold {i} - confusion matrix:\n{cm}\n")
print("CNN mean precision:", np.mean([precision_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_cnn, y_pred_cnn)]))
print("CNN mean recall:", np.mean([recall_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_cnn, y_pred_cnn)]))
print("CNN mean f1:", np.mean([f1_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_cnn, y_pred_cnn)]))

# 5) Analogicznie MLP (rozpłaszczenie X przed CV)
X_flat = nn.flatten_data(X, img_type='color')
model_mlp = nn.create_model_mlp(
    n_classes=n_classes,
    input_shape=(X_flat.shape[1],),
    optimizer='adam',
    func_activation='relu',
    kernel_initializer='he_uniform'
)
scores_mlp, y_true_mlp, y_pred_mlp = nn.cross_validation(
    n_splits=5,
    X=X_flat, y=np.argmax(y, axis=1),
    model=model_mlp,
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)
print("MLP fold scores (accuracy):", scores_mlp, "mean:", np.mean(scores_mlp))

# Metryki dla MLP
for i, (yt, yp) in enumerate(zip(y_true_mlp, y_pred_mlp), 1):
    prec = precision_score(yt, yp, average='weighted', zero_division=0)
    rec = recall_score(yt, yp, average='weighted', zero_division=0)
    f1 = f1_score(yt, yp, average='weighted', zero_division=0)
    cm = confusion_matrix(yt, yp)
    print(f"Fold {i} - precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")
    print(f"Fold {i} - confusion matrix:\n{cm}\n")
    print("MLP mean precision:", np.mean([precision_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_mlp, y_pred_mlp)]))
    print("MLP mean recall:", np.mean([recall_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_mlp, y_pred_mlp)]))
    print("MLP mean f1:", np.mean([f1_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_mlp, y_pred_mlp)]))

with open("results_4_5.txt", "w", encoding="utf-8") as f:
    f.write("==== CNN ====" + "\n")
    f.write(f"CNN fold scores (accuracy): {scores_cnn}, mean: {np.mean(scores_cnn):.4f}\n")
    for i, (yt, yp) in enumerate(zip(y_true_cnn, y_pred_cnn), 1):
        prec = precision_score(yt, yp, average='weighted', zero_division=0)
        rec = recall_score(yt, yp, average='weighted', zero_division=0)
        f1 = f1_score(yt, yp, average='weighted', zero_division=0)
        f.write(f"Fold {i} - precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}\n")
    f.write(f"CNN mean precision: {np.mean([precision_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_cnn, y_pred_cnn)]):.4f}\n")
    f.write(f"CNN mean recall: {np.mean([recall_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_cnn, y_pred_cnn)]):.4f}\n")
    f.write(f"CNN mean f1: {np.mean([f1_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_cnn, y_pred_cnn)]):.4f}\n\n")
    f.write("==== MLP ====" + "\n")
    f.write(f"MLP fold scores (accuracy): {scores_mlp}, mean: {np.mean(scores_mlp):.4f}\n")
    for i, (yt, yp) in enumerate(zip(y_true_mlp, y_pred_mlp), 1):
        prec = precision_score(yt, yp, average='weighted', zero_division=0)
        rec = recall_score(yt, yp, average='weighted', zero_division=0)
        f1 = f1_score(yt, yp, average='weighted', zero_division=0)
        f.write(f"Fold {i} - precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}\n")
    f.write(f"MLP mean precision: {np.mean([precision_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_mlp, y_pred_mlp)]):.4f}\n")
    f.write(f"MLP mean recall: {np.mean([recall_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_mlp, y_pred_mlp)]):.4f}\n")
    f.write(f"MLP mean f1: {np.mean([f1_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(y_true_mlp, y_pred_mlp)]):.4f}\n")

with open("results_4_6_macierze.txt", "w", encoding="utf-8") as f:
    f.write("==== CNN ====" + "\n")
    for i, (yt, yp) in enumerate(zip(y_true_cnn, y_pred_cnn), 1):
        cm = confusion_matrix(yt, yp)
        f.write(f"Fold {i} - macierz pomyłek:\n{cm}\n\n")
    f.write("==== MLP ====" + "\n")
    for i, (yt, yp) in enumerate(zip(y_true_mlp, y_pred_mlp), 1):
        cm = confusion_matrix(yt, yp)
        f.write(f"Fold {i} - macierz pomyłek:\n{cm}\n\n")

# 3) Przygotuj model CNN z publikacji i callback EarlyStopping
n_classes = y.shape[1]
input_shape = X.shape[1:]  # (H, W, C)
model_cnn_ear = nn.create_model_cnn_ear_biometrics(
    n_classes=n_classes,
    input_shape=input_shape,
    optimizer='adam'
)
callbacks = [EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]

# 4) 5-krotna walidacja dla modelu z publikacji, 500 epok, wyniki do CSV
e_scores, e_y_true, e_y_pred = nn.cross_validation(
    n_splits=5,
    X=X, y=y,
    model=model_cnn_ear,
    epochs=500,
    batch_size=16,
    callbacks=callbacks,
    csv_path="results_5_ear_biometrics.csv"
)
with open("results_5_ear_biometrics.txt", "w", encoding="utf-8") as f:
    f.write("==== CNN (architektura z publikacji) ====" + "\n")
    f.write(f"Fold scores (accuracy): {e_scores}, mean: {np.mean(e_scores):.4f}\n")
    for i, (yt, yp) in enumerate(zip(e_y_true, e_y_pred), 1):
        prec = precision_score(yt, yp, average='weighted', zero_division=0)
        rec = recall_score(yt, yp, average='weighted', zero_division=0)
        f1 = f1_score(yt, yp, average='weighted', zero_division=0)
        cm = confusion_matrix(yt, yp)
        f.write(f"Fold {i} - precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}\n")
        f.write(f"Fold {i} - confusion matrix:\n{cm}\n\n")
    f.write(f"Mean precision: {np.mean([precision_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(e_y_true, e_y_pred)]):.4f}\n")
    f.write(f"Mean recall: {np.mean([recall_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(e_y_true, e_y_pred)]):.4f}\n")
    f.write(f"Mean f1: {np.mean([f1_score(yt, yp, average='weighted', zero_division=0) for yt, yp in zip(e_y_true, e_y_pred)]):.4f}\n")
