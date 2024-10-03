import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Download NLTK data (indien nodig)
nltk.download('punkt')
nltk.download('stopwords')

# Uitgebreidere lijst met rapportages (dagverslagen van de cliënt)
reports = [
    "Vandaag voelde ik me goed, geen pijn of koorts.",
    "Ik heb last van hoofdpijn en voelde me erg moe.",
    "Veel koorts vandaag, mijn lichaam doet pijn.",
    "Een goede dag, ik had geen klachten.",
    "Erge pijn en koorts vandaag, voelde me verschrikkelijk.",
    "Ik was de hele dag moe en had last van mijn rug.",
    "Voelde me energiek vandaag, kon veel doen zonder problemen.",
    "Ik had een beetje hoofdpijn, maar verder ging het goed.",
    "Vandaag erg duizelig en misselijk, moeilijk om iets te doen.",
    "Ik had vandaag geen klachten, voelde me fit en gezond.",
    "Hele dag last van buikpijn en heb bijna niet kunnen eten.",
    "Een rustige dag zonder problemen, voelde me ontspannen.",
    "Vanochtend last van rillingen, maar het ging beter in de middag.",
    "Zeer vermoeiend, kon niet veel doen door mijn rugpijn.",
    "Voelde me positief, had geen last van klachten.",
    "Vandaag was het zwaar, voelde me heel vermoeid en misselijk.",
    "Ik heb vandaag genoten van een wandeling, voelde me gezond.",
    "Voelde me onrustig en had moeite om in slaap te vallen.",
    "Hoofdpijn en koorts, het was moeilijk om op te staan.",
    "Een zeer goede dag, ik was energiek en kon zonder problemen werken.",
    "Vandaag erg misselijk, heb de hele dag in bed gelegen.",
    "Lichte vermoeidheid, maar over het algemeen voelde ik me goed.",
    "Hele dag last van kortademigheid en duizeligheid.",
    "Voelde me emotioneel en erg gestrest vandaag.",
    "Een heerlijke dag zonder klachten, ik had veel energie.",
    "Erg veel last van migraine, ik kon niets doen vandaag.",
    "Voelde me kalm en ontspannen, het was een goede dag.",
    "Veel hoesten en rillingen, ik ben de hele dag thuis gebleven.",
    "Slecht geslapen, voelde me daardoor de hele dag futloos.",
    "Kon goed ademen en had geen last van pijn, een goede dag.",
    "Veel last van rugpijn, het was een zware dag.",
    "Voelde me vandaag weer energiek en kon goed bewegen.",
    "De hele dag benauwd en hoofdpijn, het was moeilijk om iets te doen.",
    "Het ging redelijk goed vandaag, had maar een beetje pijn."
]

# Labels die aangeven of de dag goed of slecht was (1 = Goed, 0 = Slecht)
labels = [
    1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 
    0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
    1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
    1, 0, 1, 0
]

# Maak een DataFrame
data = {
    'report': reports,
    'label': labels
}

# Creëer de DataFrame
df = pd.DataFrame(data)

# Opslaan als CSV-bestand
df.to_csv('rapportages_uitgebreid.csv', index=False, encoding='utf-8')

# Bevestiging
print("Uitgebreide DataFrame opgeslagen als 'rapportages_uitgebreid.csv'")

# Preprocessing: tokenisatie en stopwoorden verwijderen
stop_words = set(stopwords.words('dutch'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

# Preprocess alle rapporten
processed_reports = [preprocess(report) for report in reports]

# Vectoriseer de tekst met TF-IDF (omzetten van tekst naar numerieke vectoren)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_reports)

# Split de data in een training en test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model: Gebruik Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Voorspel op de test set
y_pred = model.predict(X_test)

# Evaluatie: Bereken de nauwkeurigheid - wordt volgens mij niet geprint
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Gedetailleerde classificatierapport
print(classification_report(y_test, y_pred))

# Confusion Matrix - hieruit komt dat het model dagen negatief labelt terwijl ze positief waren
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Optioneel: Voorspel voor alle rapportages en toon ze met de bijbehorende voorspellingen
predicted_scores = model.predict(X)

# Visualisatie van de voorspellingen
days = np.arange(1, len(reports) + 1)
plt.plot(days, predicted_scores, marker='o')
plt.title('Dagelijkse gezondheidsrating voorspeld door Naive Bayes')
plt.xlabel('Dag')
plt.ylabel('Gezondheidsrating (0=Slecht, 1=Goed)')
plt.xticks(days)
plt.grid(True)
plt.show()

# Toon de voorspellingen per dag - !print nog niet volgens mij!
for i, score in enumerate(predicted_scores):
    print(f"Dag {i+1}: Rapportage: '{reports[i]}' - Voorspelde gezondheidsrating: {score}")
