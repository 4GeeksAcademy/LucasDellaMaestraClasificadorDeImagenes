
# Paso 1: Importar librerías
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Input
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Paso 2: Cargar y filtrar CIFAR-10 (clases: avión=0, barco=8)
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test_full = y_test_full.flatten()
train_filter = np.where((y_train_full == 0) | (y_train_full == 8))
test_filter = np.where((y_test_full == 0) | (y_test_full == 8))
X_train, y_train = X_train_full[train_filter], y_train_full[train_filter]
X_test, y_test = X_test_full[test_filter], y_test_full[test_filter]

# Paso 3: Normalizar imágenes y binarizar etiquetas
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np.where(y_train == 0, 0, 1)
y_test = np.where(y_test == 0, 0, 1)

# Paso 4: Definir modelo CNN simple
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 5: Entrenar modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Paso 6: Evaluar rendimiento
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Precisión final en test: {test_acc:.4f} | Pérdida: {test_loss:.4f}')

# Paso 7: Curva ROC y AUC
y_probs = model.predict(X_test).flatten()
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.grid(True)
plt.show()

# Paso 8: Mostrar predicciones sobre imágenes reales
class_names = ['Avión', 'Barco']
indices = np.random.choice(len(X_test), 9, replace=False)
images = X_test[indices]
true_labels = y_test[indices]
pred_probs = model.predict(images).flatten()
pred_labels = (pred_probs > 0.5).astype(int)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.axis('off')
    plt.title(f'Real: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}')
plt.tight_layout()
plt.show()

# Paso 9: Guardar modelo en formato moderno .keras
model.save("modelo_avion_barco.keras")