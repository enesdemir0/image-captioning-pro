import tensorflow as tf
import yaml
import mlflow

def test_imports():
    """Ensure all critical libraries are installed and importable."""
    assert tf.__version__ is not None
    assert yaml.__version__ is not None
    assert mlflow.__version__ is not None

def test_tensorflow_basic_math():
    """Simple check to see if TensorFlow can perform a basic operation."""
    a = tf.constant(5)
    b = tf.constant(10)
    assert (a + b).numpy() == 15