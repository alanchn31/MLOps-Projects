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

## Serve Model:
- Serve the Models with Local REST server
`mlflow models serve -m runs:/<RUN_ID>/model --port 9000`

`mlflow models serve -m file:///C:/Users/alanc/OneDrive/Desktop/MLOps-Projects/1.%20Experiment%20Tracking/mlruns/311895239248179124/33d687362a4d4e329167f2db712e26e7/artifacts/model_pipeline --port 9000`

- Test the API by sending in a JSON using POST request to `http://127.0.0.1:9000/invocations`:
```bash
{
    "dataframe_split": {
        "columns": [
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
            "avg_glucose_level",
            "bmi",
            "age"
        ],
        "data": [
            [
                0,
                0,
                "Yes",
                "Govt_job",
                "Urban",
                "never smoked",
                95.94,
                31.1,
                30
            ]
        ]
    }
}
```

- We can also invoke the API using curl:
```bash
curl --location 'http://127.0.0.1:9000/invocations' \
--header 'Content-Type: application/json' \
--data '{
    "dataframe_split": {
        "columns": [
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
            "avg_glucose_level",
            "bmi",
            "age"
        ],
        "data": [
            [
                0,
                0,
                "Yes",
                "Govt_job",
                "Urban",
                "never smoked",
                95.94,
                31.1,
                30
            ]
        ]
    }
}'
```

