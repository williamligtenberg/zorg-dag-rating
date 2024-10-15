import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Download NLTK data (indien nodig)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# Combineer alle woorden in één woordenboek
word_scores = {**negative_word_scores, **positive_word_scores}

# Voorbeeldrapportages (dagverslagen van de cliënt)
reports = [
    "Vandaag voelde ik me goed, geen pijn of koorts.",
    "Ik heb last van hoofdpijn en voelde me erg moe.",
    "Veel koorts vandaag, mijn lichaam doet pijn.",
    "Een goede dag, ik had geen klachten.",
    "Erge pijn en koorts vandaag, voelde me verschrikkelijk."
]

# Preprocessing: stopwoorden verwijderen, lemmatisering, tokenisatie
stop_words = set(stopwords.words('dutch'))  # Stopwoorden voor Nederlands
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

# Functie om de score voor een rapport te berekenen
def calculate_report_score(report):
    tokens = preprocess(report)
    score = 0
    for token in tokens:
        if token in word_scores:
            score += word_scores[token]
    return score

# Bereken de scores voor alle rapportages
daily_scores = [calculate_report_score(report) for report in reports]

# Normaliseer de scores (omgekeerd: hoge score = goede dag)
def normalize_scores(scores):
    max_score = max(scores)  # Bepaal de maximale score
    min_score = min(scores)  # Bepaal de minimale score
    normalized = [(score - min_score) / (max_score - min_score) * 10 for score in scores]
    return [10 - score for score in normalized]  # Draai de score om: hoge score = goede dag

normalized_scores = normalize_scores(daily_scores)

# Visualisatie van de resultaten
days = np.arange(1, len(reports) + 1)

plt.plot(days, normalized_scores, marker='o')
plt.title('Dagelijkse gezondheidsrating van de cliënt')
plt.xlabel('Dag')
plt.ylabel('Gezondheidsrating (1-10)')
plt.xticks(days)
plt.grid(True)
plt.show()

# Resultaten per dag weergeven
for i, score in enumerate(normalized_scores):
    print(f"Dag {i+1}: Rapportage: '{reports[i]}' - Gezondheidsrating: {score:.2f}")