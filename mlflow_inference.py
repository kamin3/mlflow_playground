import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

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
    scorer = ModelScorer("LogisticRegressionModel", 1)
    sample_input = {
        "sepal length (cm)": [5.1, 7.0],
        "sepal width (cm)": [3.5, 3.2],
        "petal length (cm)": [1.4, 4.7],
        "petal width (cm)": [0.2, 1.4]
    }
    predictions = scorer.score(sample_input)
    print("Predictions:", predictions)
