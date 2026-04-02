import pytest
import tensorflow as tf
from src.utils.config_loader import load_config
from src.models.encoder import CNN_Encoder

@pytest.fixture
def config():
    """Fixture to load the configuration for testing."""
    return load_config()

def test_encoder_output_shape(config):
    """
    Test that the Encoder takes an image and produces 
    the correct feature shape.
    """
    # 1. Initialize the Encoder
    encoder = CNN_Encoder(config)
    embedding_dim = config['model']['embedding_dim']
    
    # 2. Create a "Fake" Image batch (Batch Size=2, Height=299, Width=299, Channels=3)
    # Note: If your config uses VGG16, change 299 to 224
    if config['model']['encoder_name'] == "InceptionV3":
        input_shape = (2, 299, 299, 3)
    else:
        input_shape = (2, 224, 224, 3)
        
    sample_input = tf.random.uniform(input_shape)

    # 3. Pass through the encoder
    output = encoder(sample_input)

    # 4. ASSERTIONS: The "Pro" way to check results
    # The output should have 3 dimensions: (Batch, Regions, EmbeddingDim)
    assert len(output.shape) == 3
    assert output.shape[0] == 2             # Batch size must match
    assert output.shape[2] == embedding_dim # Embedding dimension must match config
    
    print(f"\n✅ Encoder Output Shape: {output.shape}")

def test_encoder_invalid_name():
    """Test that the Encoder raises an error for a bad model name."""
    bad_config = {
        'model': {
            'encoder_name': "SuperAIBrain9000",
            'embedding_dim': 256
        }
    }
    with pytest.raises(ValueError):
        CNN_Encoder(bad_config)