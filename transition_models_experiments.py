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
    
    # For Image Classification Model
    image_classification_version = 1  # Use the correct version number
    transitioner.transition_to_production("ImageClassificationModel", image_classification_version)
    
    # For House Price Prediction Model
    house_price_prediction_version = 1  # Use the correct version number
    transitioner.transition_to_production("HousePricePredictionModel", house_price_prediction_version)
