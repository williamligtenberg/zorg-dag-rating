import pandas as pd
import sqlite3

# Stap 1: Laad de CSV-data in een Pandas DataFrame
df = pd.read_csv("zorgdata.csv")

# Stap 2: Maak verbinding met (of maak) een SQLite-database
conn = sqlite3.connect("rapportages.db")
cursor = conn.cursor()

# Stap 3: Maak een nieuwe tabel 'rapporten' met een auto-increment 'id'-kolom
cursor.execute('''
CREATE TABLE IF NOT EXISTS rapporten (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report TEXT,
    score INTEGER
)
''')

# Stap 4: Sla de data van de DataFrame op in de tabel, zonder de ID-kolom in te voeren
df.to_sql("rapporten_temp", conn, if_exists="replace", index=False)

# Stap 5: Kopieer gegevens van tijdelijke tabel naar definitieve tabel en geef ID's automatisch toe
cursor.execute('''
INSERT INTO rapporten (report, score)
SELECT report, score FROM rapporten_temp
''')

# Stap 6: Verwijder de tijdelijke tabel
cursor.execute("DROP TABLE rapporten_temp")

# Stap 7: Sla wijzigingen op en sluit de verbinding
conn.commit()
conn.close()

print("Data uit zorgdata.csv is succesvol geïmporteerd met unieke ID's.")