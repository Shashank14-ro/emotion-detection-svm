# emotion-detection-svm
Classical ML (SVM) approach on extracted facial feature

# Emotion Detection using SVM 😊📊

This project performs facial emotion classification using a Support Vector Machine (SVM). It uses extracted features from facial images and trains a classical machine learning model to classify emotions such as happy, sad, angry, and surprised.


## 🧠 Features

- Facial emotion classification using SVM
- Feature extraction from facial landmarks or image pixels
- Trained and tested on emotion datasets (e.g., FER2013, JAFFE)
- High accuracy with classical ML approach
- Lightweight and fast for small applications


## 💻 Tech Stack

- **Language**: Python
- **Libraries**:
  - OpenCV (for image handling and face detection)
  - Scikit-learn (SVM, preprocessing)
  - NumPy, pandas, matplotlib
- **Optional**: dlib or Mediapipe (for facial landmarks)
```
emotion-detection-svm/
├── dataset/ # Dataset images or CSVs
│ └── fer2013.csv
├── models/ # Trained SVM model (pickle file)
│ └── svm_model.pkl
├── notebooks/ # Jupyter notebooks for analysis
│ └── train_svm.ipynb
├── src/ # Python scripts
│ ├── feature_extraction.py
│ ├── train.py
│ ├── predict.py
│ └── utils.py
├── requirements.txt
├── README.md
└── LICENSE
```
🔬 Feature Extraction Methods
```
Option 1: Flattened grayscale pixel values (basic)

Option 2: Facial landmarks (using dlib or Mediapipe)

Option 3: Histogram of Oriented Gradients (HOG)
```
You can switch the method in feature_extraction.py.
📈 Results
```
Emotion	Accuracy (%)
Happy	89%
Sad	85%
Angry	83%
Surprise	91%
```
