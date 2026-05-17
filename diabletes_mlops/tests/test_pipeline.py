import pytest
import numpy as np
import joblib, sys, os

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath('.'))
from src.data_loader import preprocess_data, build_preprocessor


def build_test_pipeline(X_train, feature_names):
    preprocessor = build_preprocessor(feature_names)
    return Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])


class TestPipelineStructure:
    def test_pipeline_has_two_steps(self, split_data):
        X_train, _, _, _, feature_names = split_data
        pipeline = build_test_pipeline(X_train, feature_names)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'preprocessing'
        assert pipeline.steps[1][0] == 'classifier'
    

    def test_pipeline_predict_returns_binary(self, split_data):
        """predict() doit retourner uniquement des 0 et des 1."""
        X_train, X_test, y_train, _, feature_names = split_data
        pipeline = build_test_pipeline(X_train, feature_names)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert set(np.unique(y_pred)).issubset({0, 1}), \
        f'predict() retourne des valeurs inattendues : {set(np.unique(y_pred))}'

    

    def test_pipeline_predict_proba_shape(self, split_data):
        """predict_proba() : shape (n_samples, 2), probas sommant à 1."""
        X_train, X_test, y_train, _, feature_names = split_data
        pipeline = build_test_pipeline(X_train, feature_names)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(y_proba.sum(axis=1), np.ones(len(X_test)), atol=1e-6)

    
    def test_pipeline_saved_includes_preprocessor(self, split_data, tmp_path):
        """Le modèle sauvegardé doit être le PIPELINE complet (avec preprocessing)."""
        X_train, _, y_train, _, feature_names = split_data
        pipeline = build_test_pipeline(X_train, feature_names)
        pipeline.fit(X_train, y_train)
        model_path = tmp_path / 'test_model.pkl'
        joblib.dump(pipeline, model_path)

        loaded = joblib.load(model_path)
        # Détecte le Bug 5 : sauvegarde du classifieur seul
        assert isinstance(loaded, Pipeline), \
        f'Chargé type={type(loaded)}, attendu Pipeline (bug sauvegarde)'
        assert 'preprocessing' in loaded.named_steps

    


class TestNoDataLeakage:
    """Vérifie l'absence de fuite de données train → test."""

    def test_scaler_fitted_only_on_train(self, split_data):
        """Les stats du scaler doivent correspondre à X_train uniquement."""
        X_train, X_test, y_train, _, feature_names = split_data
        pipeline = build_test_pipeline(X_train, feature_names)
        pipeline.fit(X_train, y_train)
        scaler = (pipeline.named_steps['preprocessing']
                                        .named_transformers_['num']
                                        .named_steps['scaler'])
        for i in range(X_train.shape[1]):
            expected = X_train[:, i].mean()
            actual = scaler.mean_[i]
            assert abs(actual - expected) < 1e-4, \
            f'{feature_names[i]}: mean scaler={actual:.4f} != X_train={expected:.4f}'
            \
            ' — data leakage possible'