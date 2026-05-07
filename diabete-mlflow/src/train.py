import os, sys, joblib
import time
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_diabetes_dataset, preprocess_data, build_preprocessor
from src.evaluate import compute_metrics, plot_confusion_matrix
from config import MODELS_DIR, RANDOM_STATE, MLFLOW_EXPERIMENT, MODEL_NAME

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# MLFlow configuration
mlflow.set_experiment(MLFLOW_EXPERIMENT)

def train_and_track(
    n_estimators: int = 100,
    max_depth = None,
    min_samples_split: int = 2,
    min_samples_leaf:int = 1, 
    class_weight= None,
    run_name: str= 'RF_baseline'
    ):

    with mlflow.start_run(run_name = run_name) as run:    
        # tags for the current run
        mlflow.set_tags({
        'algorithm' : 'RandomForestClassifier',
        'dataset': 'Pima Indians Diabetes',
        'task': 'binary_classification',
        'framework' : 'scikit-learn',
        })

        # 1. Chargement des donnees
        df = load_diabetes_dataset()
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        preprocessor = build_preprocessor(feature_names)
        
        # hyperparams logging
        mlflow.log_params(params= {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth if max_depth else 'unlimited',
                                'min_samples_split' : min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'class_weight': str(class_weight),
                                'random_state': RANDOM_STATE,
                                'test_size': 0.2,
                                'train_samples': len(X_train),
                                'test_samples': len(X_test),
                                'n_features': len(feature_names),
                                })

        # 2. Pipeline complet
        pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        class_weight=class_weight,
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1
                                        ))
        ])

        # 3. Entrainement
        print(f'Entrainement : {run_name}')
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # 4. Enregistrement des METRIQUES (train ET test)
        y_pred= pipeline.predict(X_test)
        y_proba= pipeline.predict_proba(X_test)[:, 1]
        test_metrics= compute_metrics(y_test, y_pred, y_proba)

        y_train_pred= pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)

        mlflow.log_metrics({f'test_{k}' : v for k, v in test_metrics.items()})
        mlflow.log_metrics({f'train_{k}': v for k, v in train_metrics.items()})
        mlflow.log_metric('training_time_seconds', round(training_time, 3))

        # 7. Enregistrement du MODELE avec signature
        input_example = pd.DataFrame(X_test[:3], columns=feature_names)
        output_example = pd.DataFrame(y_pred[:3], columns=['Outcome'])
        signature = infer_signature(input_example, output_example)
        mlflow.sklearn.log_model(
                                pipeline,
                                name='model',
                                serialization_format="skops",
                                signature=signature,
                                input_example=input_example,
                                registered_model_name=MODEL_NAME # -> Model Registry
                                )
        
        # 8. Artefacts visuels — matrice de confusion
        cm_path = plot_confusion_matrix(y_test, y_pred, run_name, MODELS_DIR)
        mlflow.log_artifact(cm_path, artifact_path='plots')

        # 9. Rapport texte
        report = classification_report(y_test, y_pred)
        report_path = os.path.join(MODELS_DIR, f'report_{run_name}.txt')
        with open(report_path, 'w') as f:
            f.write(f'Run: {run_name}\n\n' + report)
        mlflow.log_artifact(report_path, artifact_path='reports')
        print(f'Run "{run_name}" termine | ID: {run.info.run_id[:8]}...')
        print(f'test_roc_auc={test_metrics["roc_auc"]:.4f}')
        print(f'train_roc_auc={train_metrics["roc_auc"]:.4f}')
        return pipeline, test_metrics, run.info.run_id
    

if __name__ == '__main__':
    train_and_track(n_estimators=100, max_depth=None, run_name='RF_baseline')

