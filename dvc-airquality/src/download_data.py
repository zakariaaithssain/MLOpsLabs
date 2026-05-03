# src/download_data.py
import urllib.request, zipfile, os, sys
import pandas as pd
URL = 'https://archive.ics.uci.edu/static/public/360/air+quality.zip'
ZIP_DST = 'data/raw/air_quality.zip'
OUT_DIR = 'data/raw'
os.makedirs(OUT_DIR, exist_ok=True)
# Téléchargement
if not os.path.exists(ZIP_DST):
    print(f'Téléchargement : {URL}')
    try:
        urllib.request.urlretrieve(URL, ZIP_DST)
        print(f' ZIP sauvegardé : {ZIP_DST}')
    except Exception as e:
        print(f' Erreur : {e}')
        sys.exit(1)
else:
    print(f'ZIP déjà présent : {ZIP_DST}')
# Décompression
print('Décompression...')
with zipfile.ZipFile(ZIP_DST) as z:
    z.extractall(OUT_DIR)
    print(f' Fichiers extraits : {z.namelist()}')
# Vérification
csv_path = os.path.join(OUT_DIR, 'AirQualityUCI.csv')
df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='latin-1')
df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
print(f'\n--- Vérification ---')
print(f'Lignes : {df.shape[0]} (attendu : ~9358)')
print(f'Colonnes: {df.shape[1]} (attendu : 15)')
print(f'Colonnes: {list(df.columns)}')
miss = (df == -200).sum().sum()
print(f'Valeurs -200 (manquantes) : {miss}')
