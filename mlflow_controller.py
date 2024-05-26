import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

class ModelMetricsRetriever:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = mlflow.tracking.MlflowClient()

    def get_production_version(self):
        # Get all versions of the model
        versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if versions:
            # Assuming there's only one version in Production stage
            production_version = versions[0]
            return production_version.version
        else:
            raise ValueError(f"No versions of model {self.model_name} found in Production stage.")

    def get_metrics(self, model_version):
        # Get the run ID associated with the model version
        model_version_details = self.client.get_model_version(self.model_name, model_version)
        run_id = model_version_details.run_id

        # Get the metrics logged in that run
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        return metrics

# Example usage
if __name__ == "__main__":
    model_name = "RandomForestModel"  # Replace with your model name
    retriever = ModelMetricsRetriever(model_name)

    try:
        production_version = retriever.get_production_version()
        print(f"Model '{model_name}' is in Production with version: {production_version}")

        metrics = retriever.get_metrics(production_version)
        print(f"Metrics for model version {production_version}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")

    except ValueError as e:
        print(e)
