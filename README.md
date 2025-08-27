# Iris Classifier (Decision Tree)

## Overview
End-to-end ML example: builds a decision-tree classifier on the Iris dataset using scikit-learn.

## Quick start
```bash
git clone https://github.com/<YOUR_USERNAME>/iris-classifier.git
cd iris-classifier
python -m venv venv && .\venv\Scripts\activate  # Windows
pip install -r requirements.txt
python src/train.py --test-size 0.2 --random-state 42

iris-classifier/
├── data/
├── notebooks/
│   └── iris_model.ipynb
├── outputs/
│   ├── confusion_matrix.png
│   └── model.joblib
├── src/
│   └── train.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
