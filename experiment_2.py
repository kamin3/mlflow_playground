import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

# Define the experiment for house price prediction
mlflow.set_experiment("House Price Prediction Experiment")

# Load the California housing dataset
california = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

# Train and log the model
with mlflow.start_run(run_name="HousePricePredictionModelRun") as run:
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Log the model
    mlflow.sklearn.log_model(model, "house_price_prediction_model")
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/house_price_prediction_model"
    model_details = mlflow.register_model(model_uri, "HousePricePredictionModel")
    
    print(f"Model 'HousePricePredictionModel' registered with accuracy: {accuracy}")
