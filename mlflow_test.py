import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Set the tracking URI to the local MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

# Load dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

# Train a model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Infer the signature of the model
signature = infer_signature(X_train, model.predict(X_train))

# Start a new run and log the model
with mlflow.start_run(run_name="GradientBoostingModel") as run:
    mlflow.sklearn.log_model(model, "gradient_boosting_model", signature=signature)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/gradient_boosting_model"
    model_name = "GradientBoostingModel"
    mlflow.register_model(model_uri, model_name)

    print(f"Model registered as '{model_name}' with URI '{model_uri}'")

# Transition the model to "Production" stage
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_version_info = client.get_latest_versions(model_name, stages=["None"])
latest_version = latest_version_info[0].version
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production"
)

print(f"Model '{model_name}' version {latest_version} transitioned to 'Production' stage")
