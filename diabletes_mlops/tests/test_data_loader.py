import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('.'))

from src.data_loader import preprocess_data, build_preprocessor
from config import TARGET_COLUMN



class TestZeroReplacement:
    """Vérifie que les zéros médicalement impossibles sont remplacés."""

    def test_no_nan_after_preprocessing(self, df_with_zeros):
        """Aucune valeur NaN ne doit subsister après preprocessing."""
        X_train, X_test, _, _, _ = preprocess_data(df_with_zeros.copy())
        assert not np.isnan(X_train).any(), 'NaN dans X_train après preprocessing'
        assert not np.isnan(X_test).any(), 'NaN dans X_test après preprocessing'

    
    def test_glucose_no_zeros(self, df_with_zeros):
        """Les zéros dans Glucose doivent être remplacés par la médiane."""
        X_train, X_test, _, _, feature_names = preprocess_data(df_with_zeros.copy())
        glucose_idx = feature_names.index('Glucose')
        all_glucose = np.concatenate([X_train[:, glucose_idx],X_test[:, glucose_idx]])
        assert (all_glucose > 0).all(), 'Des zéros persistent dans Glucose'

    
    def test_output_types_are_numeric(self, df_with_zeros):
        """Les tableaux retournés doivent être de type float."""
        X_train, _, _, _, _ = preprocess_data(df_with_zeros.copy())
        assert X_train.dtype in [np.float32, np.float64], f'X_train dtype inattendu : {X_train.dtype}'

    

class TestTrainTestSplit:
    """Vérifie les propriétés statistiques du split train/test."""

    def test_split_sizes(self, split_data):
        """80% train / 20% test."""
        X_train, X_test, y_train, y_test, _ = split_data
        total = len(X_train) + len(X_test)
        assert abs(len(X_train) / total - 0.8) < 0.02, \
        f'Ratio train attendu 0.80, obtenu {len(X_train)/total:.2f}'

    
    def test_stratified_split_class_balance(self, sample_df):
        """Le ratio de classes doit être préservé entre train et test (stratify)."""
        from src.data_loader import preprocess_data
        df = sample_df.copy()
        df['Outcome'] = 0
        df.loc[:79, 'Outcome'] = 1
        # exactement 40%
        _, _, y_train, y_test, _ = preprocess_data(df)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05, \
        f'Desequilibre : train={train_ratio:.2f}, test={test_ratio:.2f}' \
        ' — verifier stratify=y dans train_test_split'


    
    def test_no_overlap_between_train_and_test(self, split_data):
        """Aucune observation dans train ET test simultanément."""
        X_train, X_test, _, _, _ = split_data
        train_set = set(map(tuple, X_train))
        test_set= set(map(tuple, X_test))
        overlap= train_set & test_set
        assert len(overlap) == 0, \
        f'{len(overlap)} observations dans train ET test (fuite potentielle)'

    
    def test_target_is_binary(self, split_data):
        """La cible doit contenir uniquement des 0 et des 1 (pas d'autres valeurs)."""
        _, _, y_train, y_test, _ = split_data
        unique = set(np.unique(y_train)) | set(np.unique(y_test))
        assert unique.issubset({0, 1}), \
        f'Valeurs cible inattendues : {unique} — verifier TARGET_COLUMN'

    
    