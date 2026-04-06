import pytest
import tensorflow as tf
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder

def test_encoder_architecture(config):
    # Tests if the Encoder produces the correct feature shape
    encoder = CNN_Encoder(config)
    sample_img = tf.random.uniform((1, 299, 299, 3))
    output = encoder(sample_img)
    
    assert len(output.shape) == 3
    # The last dimension must match the units (e.g., 512)
    assert output.shape[-1] == config['model']['units']

def test_decoder_architecture(config):
    decoder = RNN_Decoder(config)
    batch_size = 2
    units = config['model']['units']
    vocab_size = config['dataset']['vocab_size']
    
    # 1. Create dummy inputs
    dummy_input = tf.random.uniform((batch_size, 1))
    # Attention needs a grid of features (e.g., 64 spots)
    dummy_features = tf.random.uniform((batch_size, 64, units))
    
    # 2. Initialize state
    hidden = decoder.init_decoder_state(tf.random.uniform((batch_size, units)))
    
    # 3. Test forward pass
    # NEW: We now pass 3 arguments and receive 3 outputs
    output, state, attention = decoder(dummy_input, dummy_features, hidden)
    
    # 4. Assertions
    assert output.shape == (batch_size, vocab_size)
    
    # Check if attention weights exist when type is not None
    if config['model'].get('attention_type', 'None') != 'None':
        assert attention is not None
        assert len(attention.shape) == 3 # (batch, 64, 1)