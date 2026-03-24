import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Load
X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# 🔥 Normalize
if np.max(X[:, :, 1]) != 0:
    X[:, :, 1] = X[:, :, 1] / np.max(X[:, :, 1])

X[:, :, 2] = X[:, :, 2] / 1500.0

# Labels
num_classes = len(set(y))
y = to_categorical(y, num_classes)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = Sequential([
    Input(shape=(3000, 4)),

    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),

    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),

    Flatten(),

    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Training...\n")

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=4,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("\n✅ Test Accuracy:", acc)

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

print("\n📊 Confusion Matrix:\n", cm)