import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from accuracy import off_by_one_accuracy

# Data laden vanuit CSV-bestand
df = pd.read_csv('zorgdata.csv')

# CSV-bestand opslaan
df.to_csv('zorgdata.csv', index=False)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['report'], df['score'], test_size=0.2, random_state=42)

# Pipeline met SVM en TfidfVectorizer
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC())
])

# Grid Search voor hyperparameter tuning van SVM
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Training van het Grid Search model
grid.fit(X_train, y_train)

# Beste parameters weergeven
print(f"Beste parameters: {grid.best_params_}")

# Voorspellen op testdata
y_pred = grid.predict(X_test)

accuracy, off_by_one_accuracy = off_by_one_accuracy(y_test, y_pred)

print("Standard Accuracy:", accuracy)
print("Off-by-One Accuracy:", off_by_one_accuracy)

# Classificatie rapport tonen
print(classification_report(y_test, y_pred))