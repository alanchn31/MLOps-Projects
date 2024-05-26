import argparse
import mlflow
import os
import time
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

experiment_id = get_or_create_experiment("Stroke Prediction")
# Set Tracking URI (where we can access the UI)
mlflow.set_tracking_uri("http://localhost:5000")

def load_dataset():
    # load the dataset
    df = pd.read_csv("data/train.csv")
    df = df.drop('id', axis=1)
    categorical_cols = [ 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'smoking_status']
    numerical_cols = ['avg_glucose_level', 'bmi','age']
    transformer = ColumnTransformer(transformers=[('imp', SimpleImputer(strategy='median'), numerical_cols),
                                                  ('ohe', OneHotEncoder(), categorical_cols)])
    X = transformer.fit_transform(df[categorical_cols+numerical_cols])
    y = df.stroke
    RANDOM_SEED = 6
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)
    return df, X_train, X_test, y_train, y_test, transformer

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'
def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def main():
    df, X_train, X_valid, y_train, y_valid, column_transformer = load_dataset()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial):
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            }

            if params["booster"] == "gbtree" or params["booster"] == "dart":
                params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                params["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                )
            # Train XGBoost model
            bst = xgb.train(params, dtrain)
            preds = bst.predict(dvalid)
            fpr, tpr, _ = roc_curve(y_valid, preds)
            auc_score = auc(fpr, tpr)
            preds = [round(value) for value in preds]
            accuracy = accuracy_score(y_valid, preds)
            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("auc", auc_score)
            mlflow.log_metric("accuracy", accuracy)
        return auc_score

    run_name = "run_1"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True) as run:
        # Initialize the Optuna study
        study = optuna.create_study(direction="maximize")

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        study.optimize(objective, n_trials=50, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_auc", study.best_value)
       
        # Log tags
        mlflow.set_tags(
            tags={
                "project": "Stroke Prediction Project",
                "optimizer_engine": "optuna",
                "model_family": "xgboost",
                "feature_set_version": 1,
            }
        )

        # Log a fit model instance
        model = xgb.train(study.best_params, dtrain)


        clf = Pipeline(steps=[('col_transformer', column_transformer),
                              ('classifier', xgb.XGBClassifier(**study.best_params))], verbose=True)

        clf.fit(df, df['stroke'].values)

        # Log the model pipeline
        pipeline_path = 'model_pipeline'
        mlflow.sklearn.log_model(sk_model=clf, 
                                artifact_path=pipeline_path)

if __name__ == "__main__":
    main()