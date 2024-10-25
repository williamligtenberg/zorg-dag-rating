import pandas as pd
import sqlite3
import random
from datetime import datetime, timedelta

# Stap 1: Laad de CSV-data in een Pandas DataFrame
df = pd.read_csv("zorgdata.csv")

# Stap 2: Selecteer willekeurig 84 rapportages
random_rapportages = df.sample(n=84).reset_index(drop=True)

# Stap 3: Voeg een datumkolom toe
start_date = datetime(2024, 10, 18)
dates = [start_date + timedelta(days=i // 3) for i in range(84)]
random_rapportages['datum'] = dates

# Stap 4: Maak verbinding met (of maak) een SQLite-database
conn = sqlite3.connect("rapportages.db")
cursor = conn.cursor()

# Stap 5: Maak een nieuwe tabel 'rapportages' met een auto-increment 'id'-kolom en 'datum'-kolom
cursor.execute('''
CREATE TABLE IF NOT EXISTS rapportages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report TEXT,
    score INTEGER,
    datum DATE
)
''')

# Stap 6: Sla de data van de DataFrame op in de tabel, zonder de ID-kolom in te voeren
random_rapportages.to_sql("rapporten_temp", conn, if_exists="replace", index=False)

# Stap 7: Kopieer gegevens van tijdelijke tabel naar definitieve tabel en geef ID's automatisch toe
cursor.execute('''
INSERT INTO rapportages (report, score, datum)
SELECT report, score, datum FROM rapporten_temp
''')

# Stap 8: Verwijder de tijdelijke tabel
cursor.execute("DROP TABLE rapporten_temp")

# Stap 9: Sla wijzigingen op en sluit de verbinding
conn.commit()
conn.close()

print("Data uit zorgdata.csv is succesvol geïmporteerd met unieke ID's en datums.")
