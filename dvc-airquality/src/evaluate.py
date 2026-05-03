import pandas as pd, yaml, json, pickle
from sklearn.metrics import (accuracy_score, f1_score,
roc_auc_score, classification_report, confusion_matrix)

YAML_FILE = 'params.yaml'

params = yaml.safe_load(open(YAML_FILE))['evaluate']
model = pickle.load(open('models/model.pkl', 'rb'))
test = pd.read_csv('data/features/test_feat.csv')
target = 'high_pollution'

drop_cols = ['CO(GT)', target, 'Date', 'Time']
feat_cols = [c for c in test.columns if c not in drop_cols]
X_test, y_test = test[feat_cols], test[target]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

report = {
    'test_accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
    'test_f1': round(float(f1_score(y_test, y_pred)), 4),
    'test_roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
    'threshold_ok': bool(accuracy_score(y_test, y_pred) >= params['threshold']),
    'n_test': int(len(test)),
}

print(f'Résultats TEST : {report}')
if report['threshold_ok']:
    print(f"Modèle validé (accuracy >= {params['threshold']})")
else:
    print(f"Modèle rejeté (accuracy < {params['threshold']})")
    
json.dump(report, open('metrics/report.json', 'w'), indent=2)
print('\nMatrice de confusion :')
cm = confusion_matrix(y_test, y_pred)
print(f'Prédit FAIBLE | Prédit ÉLEVÉ')
print(f'Réel FAIBLE   | {cm[0,0]:5d} | {cm[0,1]:5d} |')
print(f'Réel ÉLEVÉ    | {cm[1,0]:5d} | {cm[1,1]:5d} |')
print('\n' + classification_report(
    y_test, y_pred, zero_division=0,
    target_names=['Pollution faible (CO<2)', 'Pollution élevée (CO≥2)']))
