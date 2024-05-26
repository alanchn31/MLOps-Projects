import mlflow
import pandas as pd

logged_pipeline = 'runs:/33d687362a4d4e329167f2db712e26e7/model_pipeline'
loaded_pipeline = mlflow.pyfunc.load_model(logged_pipeline)

test_data = [[0, 0, 'Yes', 'Govt_job', 'Urban', 'never smoked', 95.94, 31.1,
        30.0]]
 
categorical_cols = [ 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'smoking_status']
numerical_cols = ['avg_glucose_level', 'bmi','age']
cols = categorical_cols + numerical_cols

prediction = round(loaded_pipeline.predict(pd.DataFrame(test_data, columns=cols))[0])

print(f"Prediction is : {prediction}")