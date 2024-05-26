import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri("http://localhost:5001")


experiment_name = "Iris Classification Experiment" # we replace the experiment name with the usecase name - ecommerce-customer-churn for example
mlflow.set_experiment(experiment_name)

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# RandomForest model run
with mlflow.start_run(run_name="RandomForestModelRun") as rf_run:
    # Train a RandomForest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model with signature
    signature = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature)
    
    # Register the model
    model_uri = f"runs:/{rf_run.info.run_id}/random_forest_model"
    model_details = mlflow.register_model(model_uri, "RandomForestModel")
    
    # Transition model version to staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="RandomForestModel",
        version=model_details.version,
        stage="staging"
    )
    
    print(f"Model registered as 'RandomForestModel' with accuracy: {accuracy} and transitioned to 'staging'.")

# GradientBoosting model run
with mlflow.start_run(run_name="GradientBoostingModelRun") as gb_run:
    # Train a GradientBoosting model
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    accuracy = gb.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model with signature
    signature = infer_signature(X_train, gb.predict(X_train))
    mlflow.sklearn.log_model(gb, "gradient_boosting_model", signature=signature)
    
    # Register the model
    model_uri = f"runs:/{gb_run.info.run_id}/gradient_boosting_model"
    model_details = mlflow.register_model(model_uri, "GradientBoostingModel")
    
    # Transition model version to staging
    client.transition_model_version_stage(
        name="GradientBoostingModel",
        version=model_details.version,
        stage="staging"
    )
    
    print(f"Model registered as 'GradientBoostingModel' with accuracy: {accuracy} and transitioned to 'staging'.")

print("Both runs logged, models registered, and transitioned to staging successfully.")
