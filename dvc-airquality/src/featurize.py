import pandas as pd, yaml, os


YAML_FILE = 'src/params.yaml'

params = yaml.safe_load(open(YAML_FILE))
target_col = params['prepare']['target_col']
params = params['featurize']
EXCLUDE = [target_col, 'high_pollution', 'Date', 'Time']


def extract_time_features(df):
    """Extraire heure, jour, mois, weekend, heure de pointe depuis Date/Time."""
    if 'Date' in df.columns and 'Time' in df.columns:
        dt = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%d/%m/%Y %H.%M.%S', errors='coerce'
        )
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek

        # 0=lundi, 6=dimanche
        df['month'] = dt.dt.month
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['is_rushhour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
    return df


def process_split(path, fit_stats=None):
    df = pd.read_csv(path)
    if params['add_time_features']:
        df = extract_time_features(df)
    feat_cols = [c for c in df.columns if c not in EXCLUDE
                                        and c not in ['hour','dayofweek','month', 'is_weekend','is_rushhour']]
    
    # Supprimer lignes avec trop de NaN (seuil : moitié des features)
    if params['drop_na_rows']:
        df = df.dropna(subset=feat_cols, thresh=len(feat_cols) // 2)
        df = df.reset_index(drop=True)

    # Imputation médiane (fit sur train, appliqué sur val/test)
    if fit_stats is None:
        fit_stats = {'median': df[feat_cols].median()}
    df[feat_cols] = df[feat_cols].fillna(fit_stats['median'])

    # Features temporelles dans la liste complète
    time_feats = [c for c in ['hour','dayofweek','month', 'is_weekend','is_rushhour']
                    if c in df.columns]
    
    all_feats = feat_cols + time_feats

    # Normalisation min-max
    if params['normalize']:
        if 'min' not in fit_stats:
            fit_stats['min'] = df[all_feats].min()
            fit_stats['max'] = df[all_feats].max()
        rng = fit_stats['max'] - fit_stats['min']
        rng[rng == 0] = 1
        df[all_feats] = (df[all_feats] - fit_stats['min']) / rng
    return df, fit_stats



train, stats = process_split('data/processed/train.csv')
val,_ = process_split('data/processed/val.csv', stats)
test, _ = process_split('data/processed/test.csv', stats)

os.makedirs('data/features', exist_ok=True)
train.to_csv('data/features/train_feat.csv', index=False)
val.to_csv( 'data/features/val_feat.csv',
index=False)
test.to_csv( 'data/features/test_feat.csv', index=False)
feat_cols = [c for c in train.columns if c not in EXCLUDE]
print(f'Features prêtes — train:{len(train)}, val:{len(val)}, test:{len(test)}')
print(f'Nombre de features : {len(feat_cols)}')
print(f'Features : {feat_cols}')