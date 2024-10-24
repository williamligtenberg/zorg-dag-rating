import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from accuracy import off_by_one_accuracy
import simplemma
from flask import Flask, render_template

# Data laden vanuit CSV-bestand
df = pd.read_csv('zorgdata.csv')

# CSV-bestand opslaan
df.to_csv('zorgdata.csv', index=False)

print(df['score'].value_counts())

sws = set(stopwords.words('dutch'))

df['report'] = df['report'].apply(lambda x: ' '.join(simplemma.lemmatize(word, lang='nl') for word in x.split(' ') if word not in sws))

print(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['report'], df['score'], test_size=0.2, random_state=90)

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

#Classificatie rapport tonen
print(classification_report(y_test, y_pred))

def predict_single_value(report):
    report = ' '.join(simplemma.lemmatize(word, lang='nl') for word in report.split(' ') if word not in sws)
    # Create a DataFrame with just the report
    mdf = pd.DataFrame({'report': [report]})
    
    # Make the prediction
    y_pred = grid.best_estimator_.predict(mdf['report'])
    
    return y_pred[0]

####### HTTP SERVER
# Create a Flask app
app = Flask(__name__)

# Route for the homepage (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route for /addReport
@app.route('/addReport')
def add_report():
    return render_template('addReport.html')

if __name__ == '__main__':
    # Start the server
    app.run(debug=True)