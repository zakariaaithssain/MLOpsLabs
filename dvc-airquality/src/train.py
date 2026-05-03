import pandas as pd, yaml, os, pickle, json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


YAML_FILE =  'src/params.yaml'

params = yaml.safe_load(open(YAML_FILE))['train']
train = pd.read_csv('data/features/train_feat.csv')
val = pd.read_csv('data/features/val_feat.csv')

target = 'high_pollution'
drop_cols = ['CO(GT)', target, 'Date', 'Time']
feat_cols = [c for c in train.columns if c not in drop_cols]
X_train, y_train = train[feat_cols], train[target]
X_val,y_val = val[feat_cols], val[target]

model = RandomForestClassifier(
                        n_estimators = params['n_estimators'],
                        max_depth = params['max_depth'],
                        min_samples_split = params['min_samples_split'],
                        random_state = params['random_state'],
                        class_weight = 'balanced', # compense le déséquilibre de classes
                        n_jobs = -1 )

model.fit(X_train, y_train)
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

metrics = {
'val_accuracy' : round(float(accuracy_score(y_val, y_pred)), 4),
'val_f1'
: round(float(f1_score(y_val, y_pred)), 4),
'val_roc_auc' : round(float(roc_auc_score(y_val, y_prob)), 4),
'n_train'
: int(len(train)),
'n_features'
: int(len(feat_cols)),
}
print(f'Métriques validation : {metrics}')

# Top 5 features les plus importantes
importances = pd.Series(model.feature_importances_, index=feat_cols)
print('\nTop 5 features :')
print(importances.sort_values(ascending=False).head(5).to_string())
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
pickle.dump(model, open('models/model.pkl', 'wb'))
json.dump(metrics, open('metrics/scores.json', 'w'), indent=2)
print('\nModèle sauvegardé : models/model.pkl')