import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

# Define the experiment name (it will be created if it does not exist)
experiment_name = "Iris Classification Experiment"
mlflow.set_experiment(experiment_name)

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# RandomForest model run
with mlflow.start_run(run_name="RandomForestModelRun") as rf_run:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    mlflow.log_param("n_estimators", 100)
    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    signature = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature)
    
    model_uri = f"runs:/{rf_run.info.run_id}/random_forest_model"
    model_details = mlflow.register_model(model_uri, "RandomForestModel")
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="RandomForestModel",
        version=model_details.version,
        stage="staging"
    )
    
    print(f"Model registered as 'RandomForestModel' with accuracy: {accuracy} and transitioned to 'staging'.")

# GradientBoosting model run
with mlflow.start_run(run_name="GradientBoostingModelRun") as gb_run:
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    
    mlflow.log_param("n_estimators", 100)
    accuracy = gb.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    signature = infer_signature(X_train, gb.predict(X_train))
    mlflow.sklearn.log_model(gb, "gradient_boosting_model", signature=signature)
    
    model_uri = f"runs:/{gb_run.info.run_id}/gradient_boosting_model"
    model_details = mlflow.register_model(model_uri, "GradientBoostingModel")
    
    client.transition_model_version_stage(
        name="GradientBoostingModel",
        version=model_details.version,
        stage="staging"
    )
    
    print(f"Model registered as 'GradientBoostingModel' with accuracy: {accuracy} and transitioned to 'staging'.")

# LogisticRegression model run with custom signature
with mlflow.start_run(run_name="LogisticRegressionModelRun") as lr_run:
    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    
    mlflow.log_param("max_iter", 200)
    accuracy = lr.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Define a custom signature
    input_schema = Schema([
        ColSpec("double", "sepal length (cm)"),
        ColSpec("double", "sepal width (cm)"),
        ColSpec("double", "petal length (cm)"),
        ColSpec("double", "petal width (cm)")
    ])
    
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    mlflow.sklearn.log_model(lr, "logistic_regression_model", signature=signature)
    
    model_uri = f"runs:/{lr_run.info.run_id}/logistic_regression_model"
    model_details = mlflow.register_model(model_uri, "LogisticRegressionModel")
    
    client.transition_model_version_stage(
        name="LogisticRegressionModel",
        version=model_details.version,
        stage="staging"
    )
    
    print(f"Model registered as 'LogisticRegressionModel' with accuracy: {accuracy} and transitioned to 'staging'.")

print("All runs logged, models registered, and transitioned to staging successfully.")

class ModelScorer:
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.model = self.load_model()

    def load_model(self):
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        return model

    def score(self, input_data):
        input_df = pd.DataFrame(input_data)
        predictions = self.model.predict(input_df)
        return predictions

# Example usage
if __name__ == "__main__":
    scorer = ModelScorer("RandomForestModel", 1)
    sample_input = {
        "sepal length (cm)": [5.1, 7.0],
        "sepal width (cm)": [3.5, 3.2],
        "petal length (cm)": [1.4, 4.7],
        "petal width (cm)": [0.2, 1.4]
    }
    predictions = scorer.score(sample_input)
    print("Predictions:", predictions)

class ModelTransitioner:
    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()

    def transition_to_production(self, model_name, model_version):
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        print(f"Model {model_name} version {model_version} transitioned to production.")

# Example usage
if __name__ == "__main__":
    transitioner = ModelTransitioner()
    transitioner.transition_to_production("RandomForestModel", 1)
    transitioner.transition_to_production("GradientBoostingModel", 1)
    transitioner.transition_to_production("LogisticRegressionModel", 1)
