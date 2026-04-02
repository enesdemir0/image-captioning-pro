import tensorflow as tf
import yaml
import mlflow

if __name__ == "__main__":
  print("TensorFlow version:", tf.__version__)
  print("GPU Available:", tf.config.list_physical_devices('GPU'))
  print("MLflow and YAML are ready!")