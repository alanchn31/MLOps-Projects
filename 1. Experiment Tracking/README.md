## Project Setup
1. Create Virtual Environment with conda
```bash
conda create -n mlflow-venv python=3.10
```
2. Activate the virtual environment
```bash
conda activate mlflow-venv
```
3. Install mlflow
```bash
pip install mlflow
```
4. Launch mlflow UI
```bash
mlflow ui
```

## Running MLflow Projects:
- Create Run using MLFlow project file
`mlflow run . --experiment-name Stroke_prediction`  # run from folder where `MLProject` file is present

- Run from git repository
`mlflow run https://github.com/alanchn31/MLOps-Projects/tree/main/1.%20Experiment%20Tracking --experiment-name Stroke_prediction` 

## Test Model Prediction:
```python
python test_model.py
```