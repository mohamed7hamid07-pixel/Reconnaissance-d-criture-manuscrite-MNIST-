# ✨ Reconnaissance d'écriture manuscrite (MNIST)

## 📌 Description
Ce projet implémente un **réseau de neurones convolutionnel (CNN)** en Python avec **TensorFlow/Keras** pour reconnaître les chiffres manuscrits (0-9) à partir du dataset **MNIST**.

- Dataset : **MNIST** (60 000 images d’entraînement, 10 000 images de test).  
- Chaque image est en **niveaux de gris 28x28 pixels**.  
- Précision attendue : **98%+** après quelques epochs.

---

## 🚀 Installation
Clone le projet :
```bash
git clone https://github.com/ton-compte/handwriting_recognition.git
cd handwriting_recognition
Installe les dépendances :

bash
pip install tensorflow matplotlib

▶️ Utilisation
Lance l’entraînement :

bash
python mnist_cnn.py

Cela va :
Charger et afficher quelques exemples du dataset.
Entraîner un CNN pendant 5 epochs.
Évaluer la précision sur le jeu de test.
Sauvegarder le modèle (mnist_cnn_model.h5).
Afficher les courbes accuracy / loss.

📊 Résultats
Exemple attendu :
Accuracy train : ~99%
Accuracy test : ~98%
Les courbes montrent une bonne généralisation (pas d’overfitting excessif).

🏋️ Exercices pour améliorer le projet
Remplace le CNN par un réseau MLP et compare les résultats.

Ajoute de la data augmentation (ImageDataGenerator).

Teste ton modèle sur tes propres chiffres manuscrits.

📌 Références
Dataset MNIST
TensorFlow Documentation
