import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================
# LOAD DATA
# ======================
DATA_PATH = "landmark_data"

X = []
y = []

labels = sorted(os.listdir(DATA_PATH))

for label in labels:
    folder = os.path.join(DATA_PATH, label)
    
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)

# ======================
# PREPROCESSING
# ======================
X = X.reshape(X.shape[0], -1)

# 🔥 GLOBAL NORMALIZATION (IMPORTANT)
mean = np.mean(X)
std = np.std(X)

X = (X - mean) / (std + 1e-7)

# SAVE mean & std
np.save("mean.npy", mean)
np.save("std.npy", std)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# ======================
# MODEL
# ======================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# CALLBACKS
# ======================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

# ======================
# TRAIN
# ======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr]
)

# ======================
# SAVE
# ======================
model.save("asl_landmark_model.h5")
np.save("labels.npy", le.classes_)

print("✅ Model trained and saved!")