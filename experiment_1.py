import mlflow
import mlflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

# Define the experiment for image classification
mlflow.set_experiment("Image Classification Experiment")

# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and log the model
with mlflow.start_run(run_name="ImageClassificationModelRun") as run:
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=2)
    
    # Log the model
    mlflow.keras.log_model(model, "image_classification_model")
    
    # Log metrics
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("accuracy", accuracy)
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/image_classification_model"
    model_details = mlflow.register_model(model_uri, "ImageClassificationModel")
    
    print(f"Model 'ImageClassificationModel' registered with accuracy: {accuracy}")
