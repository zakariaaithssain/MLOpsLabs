import pytest
import numpy as np 
from numpy.random import default_rng
import pandas as pd 
import sys, os 

sys.path.insert(0, os.path.abspath('.'))


@pytest.fixture(scope='session') # use this fixture for the whole testing session.
def sample_df(): 
    """DataFrame minimal simulant le dataset Pima Indians Diabetes."""
    rng = default_rng(42)
    n = 200
    return pd.DataFrame(
        {
'Pregnancies': rng.integers(0, 10, n),
'Glucose': rng.uniform(70, 200, n),
'BloodPressure': rng.uniform(40, 120, n),
'SkinThickness': rng.uniform(0, 60, n),
'Insulin': rng.uniform(0, 300, n),
'BMI': rng.uniform(18, 50, n),
'DiabetesPedigree': rng.uniform(0.1, 2.5, n),
'Age': rng.integers(21, 80, n),
'Outcome': rng.integers(0, 2, n),
}
    )


@pytest.fixture(scope='session')
def df_with_zeros(sample_df): 
    """DataFrame avec des zéros aberrants dans les colonnes médicales."""
    df = sample_df.copy()
    df.loc[0:9, 'Glucose'] = 0 #10 zeros impossibles
    df.loc[0:4, 'BloodPressure'] = 0 # 5 zeros impossibles
    return df


@pytest.fixture(scope='session')
def split_data(sample_df): 
    """Données splitées via la fonction officielle du projet."""
    from src.data_loader import preprocess_data
    return preprocess_data(sample_df.copy())