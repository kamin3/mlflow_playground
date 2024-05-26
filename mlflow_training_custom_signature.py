import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature
from mlflow.types import Schema, ColSpec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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
