import pandas as pd, yaml, os
from sklearn.model_selection import train_test_split

YAML_FILE = 'src/params.yaml'
params = yaml.safe_load(open(YAML_FILE))['prepare']
MISSING = params['missing_value']
TARGET = params['target_col']

# Charger le CSV (séparateur ;, décimales virgule, encodage latin-1)
df = pd.read_csv('data/raw/AirQualityUCI.csv',
sep=';', decimal=',', encoding='latin-1')
print(f'Dataset brut : {df.shape[0]} lignes, {df.shape[1]} colonnes')

# Supprimer les colonnes / lignes entièrement vides (artefacts UCI)
df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')
print(f'Après nettoyage vides : {df.shape}')

# Remplacer la valeur sentinelle -200 par NaN
df = df.replace(MISSING, float('nan'))
print('Valeurs manquantes après remplacement -200→NaN :')
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# Supprimer les lignes où la cible est manquante
df = df.dropna(subset=[TARGET])
print(f'Lignes avec cible valide : {df.shape[0]}')

# Cible binaire : pollution élevée si CO(GT) >= seuil
threshold = params['co_high_threshold']
df['high_pollution'] = (df[TARGET] >= threshold).astype(int)
print(f"Pollution élevée (CO>={threshold}) : "
f"{df['high_pollution'].sum()} ({df['high_pollution'].mean()*100:.1f}%)")

# Split stratifié train / val / test
train_val, test = train_test_split(
df, test_size=params['test_size'],
random_state=params['random_state'],
stratify=df['high_pollution']
)
val_ratio = params['val_size'] / (1 - params['test_size'])
train, val = train_test_split(
train_val, test_size=val_ratio,
random_state=params['random_state'],
stratify=train_val['high_pollution']
)

os.makedirs('data/processed', exist_ok=True)
train.to_csv('data/processed/train.csv', index=False)
val.to_csv( 'data/processed/val.csv',
index=False)
test.to_csv( 'data/processed/test.csv', index=False)
print(f'Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')