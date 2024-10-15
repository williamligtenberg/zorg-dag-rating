import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (indien nodig)
nltk.download('punkt')
nltk.download('stopwords')

# Woordenlijst met negatieve woorden en hun gewicht
df = pd.read_csv('negative_word_scores.csv')
negative_word_scores = dict(zip(df['word'], df['score']))

# Positieve woorden (als je wilt dat ze een lagere score krijgen)
df = pd.read_csv('positive_word_scores.csv')
positive_word_scores = dict(zip(df['word'], df['score']))

# Uitgebreidere lijst met rapportages (dagverslagen van de cliënt)
df = pd.read_csv('rapportages.csv')
rapportages = list(df['rapportage'])  # Gebruik list om enkel de rapportages te krijgen

stop_words = set(stopwords.words('dutch'))  # Stopwoorden voor Nederlands
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

# Stap 2: Bereken de nieuwe scores
def calculate_new_score(rapportage):
    words = rapportage.split()
    total_score = 0
    
    # Tel de scores van de negatieve woorden
    for word in words:
        if word in negative_word_scores:
            total_score += negative_word_scores[word]  # Tel negatieve score op
    
    # Tel de scores van de positieve woorden
    for word in words:
        if word in positive_word_scores:
            total_score += positive_word_scores[word]  # Trek positieve score af
    
    return total_score

# Resultaten verzamelen
results = []

for rapportage in rapportages:
    new_score = calculate_new_score(rapportage)  # Bereken de nieuwe score
    # Als de score 0 is, vervang de rapportage met "neutraal"
    if new_score == 0:
        rapportage = "neutraal"
    results.append((rapportage, new_score))  # Voeg de resultaten toe aan de lijst

# Maak een DataFrame van de resultaten
results_df = pd.DataFrame(results, columns=['rapportage', 'score'])

# Sla de DataFrame op in een CSV-bestand
results_df.to_csv('rapportages_scores.csv', index=False)

print("Het CSV-bestand is succesvol aangemaakt met de nieuwe scores.")
