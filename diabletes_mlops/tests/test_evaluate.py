import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.evaluate import compute_metrics



class TestComputeMetrics:
    def test_perfect_classifier(self):
        """Un classifieur parfait doit avoir toutes ses métriques à 1.0."""
        y_true= np.array([0, 0, 0, 1, 1, 1])
        y_pred= np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics['accuracy'] == 1.0
        assert metrics['roc_auc'] == 1.0
        assert metrics['f1_score'] == 1.0


    def test_roc_auc_uses_probabilities_not_classes(self):
        """Ce test détecte le Bug 4 : roc_auc calculé sur y_pred au lieu de y_proba."""
        y_true= np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred= np.array([0, 0, 1, 1, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.85, 0.3, 0.4])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        from sklearn.metrics import roc_auc_score
        expected = roc_auc_score(y_true, y_proba)# Valeur correcte
        wrong = roc_auc_score(y_true, y_pred) # wrong value

        assert abs(metrics['roc_auc'] - expected) < 0.001, \
        f'ROC-AUC : {metrics["roc_auc"]:.4f} attendu {expected:.4f}' \
        f' (bug possible : y_pred utilisé, valeur fausse={wrong:.4f})'
    
    
    def test_metrics_values_in_range(self):
        """Toutes les métriques doivent être dans [0.0, 1.0]."""
        np.random.seed(0)
        y_true= np.random.randint(0, 2, 100)
        y_pred= np.random.randint(0, 2, 100)
        y_proba = np.random.uniform(0, 1, 100)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f'{name} hors de [0,1] : {value}'
        
    
    def test_metrics_dict_has_required_keys(self):
        """Le dict de métriques doit contenir toutes les clés requises."""
        required = {'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'}
        y_t = np.array([0, 1, 0, 1])
        y_p = np.array([0, 1, 0, 1])
        y_pr = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_metrics(y_t, y_p, y_pr)
        assert required.issubset(set(metrics.keys())), \
        f'Clés manquantes : {required - set(metrics.keys())}'

    

    