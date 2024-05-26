import mlflow
import pandas as pd

logged_transformer = 'runs:/3027451da346475dad7736fe2077e493/data_transformer'
# Load transformer
loaded_transformer = mlflow.sklearn.load_model(logged_transformer)
logged_model = 'runs:/3027451da346475dad7736fe2077e493/model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

test_data = [[0, 0, 'Yes', 'Govt_job', 'Urban', 'never smoked', 95.94, 31.1,
        30.0]]

categorical_cols = [ 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'smoking_status']
numerical_cols = ['avg_glucose_level', 'bmi','age']

X_test = loaded_transformer.transform(pd.DataFrame(test_data, columns=categorical_cols+numerical_cols))
prediction = round(loaded_model.predict(X_test)[0])
print(f"Prediction is : {prediction}")