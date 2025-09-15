"""
Projet : Reconnaissance d'écriture manuscrite avec MNIST
Auteur : Ton Nom
Date : 2025
Description : Ce script entraîne un CNN pour reconnaître les chiffres manuscrits (0-9).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==============================
# 1. Charger le dataset MNIST
# ==============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Taille train:", x_train.shape, "Taille test:", x_test.shape)

# ==============================
# 2. Visualiser quelques images
# ==============================
plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ==============================
# 3. Prétraitement
# ==============================
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape pour CNN (ajout canal = 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==============================
# 4. Définir le modèle CNN
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# ==============================
# 5. Compiler et entraîner
# ==============================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test))

# ==============================
# 6. Évaluer le modèle
# ==============================
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Précision sur le test : {test_acc*100:.2f}%")

# ==============================
# 7. Courbes d'apprentissage
# ==============================
plt.figure(figsize=(10,4))

# Courbe de la précision
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Évolution de la précision")

# Courbe de la perte
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Évolution de la perte")

plt.tight_layout()
plt.show()

# ==============================
# 8. Sauvegarder le modèle
# ==============================
model.save("mnist_cnn_model.h5")
print("📁 Modèle sauvegardé sous 'mnist_cnn_model.h5'")
