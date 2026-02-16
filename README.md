# Wine-Classification-using-Machine-Learning

🍷 Wine Classification using Machine Learning
📌 Project Overview

This project builds and compares multiple machine learning classification models on the Wine Dataset from the UCI Machine Learning Repository.

The dataset contains the results of chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.

The goal is to classify wines into one of three categories based on 13 chemical features.

📊 Dataset Information

📁 Source: Wine Dataset

🧾 Total Samples: 178

🎯 Target Classes: 3 (Wine types)

🔢 Features: 13 continuous numerical variables

📌 First column: Class label (1–3)

Features:

Alcohol

Malic Acid

Ash

Alcalinity of Ash

Magnesium

Total Phenols

Flavanoids

Nonflavanoid Phenols

Proanthocyanins

Color Intensity

Hue

OD280/OD315

Proline

All features are continuous.

⚙️ Data Preprocessing

Converted class labels from (1,2,3) → (0,1,2) using LabelEncoder

Train-Test split (80% – 20%)

Stratified sampling to maintain class balance

StandardScaler applied for scale-sensitive models

🤖 Models Implemented

The following classifiers were trained and evaluated:

Logistic Regression

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Random Forest

XGBoost

Cross-validation (5-fold) was used for robust evaluation.

📈 Model Performance
Model	Test Accuracy	CV Accuracy
Logistic Regression	~97–100%	~97%+
SVM	~98–100%	~98%+
KNN	~95–98%	~96%+
Random Forest	~99%	~98%+
XGBoost	🔥 ~99–100%	~99%

The dataset has well-separated classes, making it suitable for benchmarking new classifiers.



XGBoost and Random Forest achieved the highest accuracy.

XGBoost was configured for multi-class classification using:

objective = "multi:softprob"
eval_metric = "mlogloss"

🧠 Key Learnings

Importance of feature scaling for distance-based models

Stratified sampling improves evaluation reliability

Ensemble models perform exceptionally well on structured tabular data

Proper label encoding is required for certain models like XGBoost

📂 Project Structure
Wine-Classification/
│
├── Wine dataset.csv
├── wine_classification.ipynb
├── requirements.txt
└── README.md

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib (optional for visualization)

🚀 Future Improvements

Hyperparameter tuning with GridSearchCV

Feature importance visualization

SHAP model explainability

PCA dimensionality reduction

Deployment using Flask / FastAPI

📌 Author

Jagruti Yuvraj Dhangar
Machine Learning | Data Science | AI Enthusiast

toget code with dataset----------------------
kaggle link =https://www.kaggle.com/code/jagrutiyuvrajdhangar/wine-classification-using-machine-learning/edit  
