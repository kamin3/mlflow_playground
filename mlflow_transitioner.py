import mlflow

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


if __name__ == "__main__":
    transitioner = ModelTransitioner()
    transitioner.transition_to_production("RandomForestModel", 1)
    transitioner.transition_to_production("GradientBoostingModel", 1)
    transitioner.transition_to_production("LogisticRegressionModel", 1)
