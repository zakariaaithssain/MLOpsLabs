# run_buggy.py — démonstration : 4 bugs, aucune erreur Python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


dataset = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
df = dataset.frame.copy()
df['Outcome'] = (df['class'] == 'tested_positive').astype(int)
df = df.drop(columns=['class'])
feature_names = [c for c in df.columns if c != 'Outcome']
X = df[feature_names].values
y = df['Outcome'].values

#BUG 1 : pas de stratify
X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.2,
                                        random_state=42)

#BUG 2 : data leakage — scaler fitté sur X complet
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_s = X_scaled[:len(X_train)]
X_test_s = X_scaled[len(X_train):]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

# BUG 4 : y_pred binaire au lieu de y_proba
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC : {roc_auc:.4f}')
print('Le script s\'execute sans erreur... mais est-il correct ?')
