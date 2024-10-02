import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt

# Download NLTK data (indien nodig)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Woordenlijst met negatieve woorden en hun gewicht
negative_word_scores = {
    'koorts': 10,
    'pijn': 8,
    'vermoeid': 6,
    'misselijk': 7,
    'hoofdpijn': 5,
    'moe': 4,
    'zwakte': 7,
    'koud': 3,
    'benauwd': 9,
    'duizelig': 6,
    'griep': 10,
    'verkoudheid': 5,
    'migraine': 9,
    'slecht': 4,
    'suf': 4,
    'braken': 9,
    'buikpijn': 8,
    'diarree': 7,
    'verward': 7,
    'kortademig': 9,
    'hoesten': 6,
    'kramp': 7,
    'druk': 5,
    'zweten': 5,
    'spierpijn': 7,
    'rillingen': 6,
    'traag': 4,
    'stress': 6,
    'angst': 8,
    'slapeloosheid': 7,
    'benauwdheid': 9,
    'tintelingen': 6,
    'flauwte': 8,
    'ademnood': 9,
    'verkramping': 8,
    'kriebelhoest': 5,
    'lusteloos': 6,
    'onrustig': 5,
    'zwelling': 7,
    'duister': 5,
    'angstig': 7,
    'flauwvallen': 9,
    'bleek': 6,
    'vermoeidheid': 7,
    'onwel': 8,
    'oncomfortabel': 5,
    'tremoren': 8,
    'gebrek aan eetlust': 7,
    'spierzwakte': 7,
    'trillen': 6,
    'brandend': 7,
    'wazig zien': 6,
    'hoge bloeddruk': 8,
    'steken': 7,
    'bleekheid': 6,
    'irritatie': 5,
    'samentrekkingen': 7,
    'uitputting': 8,    
    'incontinentie': 8,        
    'val': 10,                 
    'mobiliteitsproblemen': 7, 
    'geheugenverlies': 9,      
    'dementie': 10,            
    'stijfheid': 6,            
    'depressie': 8,            
    'gewichtsverlies': 7,      
    'verwardheid': 9,          
    'slaapproblemen': 7,       
    'dorst': 5,                
    'drukplekken': 8,          
    'blaren': 6,               
    'hoge bloedsuikerspiegel': 9,
    'inactiviteit': 6,         
    'artrose': 7,              
    'onvast': 7,               
    'kramp in benen': 6,       
    'hartkloppingen': 9,       
    'verwarring': 8            
}

# Positieve woorden (als je wilt dat ze een lagere score krijgen)
positive_word_scores = {
    'gezond': -5,
    'fit': -4,
    'goed': -3,
}

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

# Normaliseer de scores (optioneel, als je een specifieke schaal wilt)
def normalize_scores(scores):
    max_score = max(scores)
    min_score = min(scores)
    return [(score - min_score) / (max_score - min_score) * 10 for score in scores]

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