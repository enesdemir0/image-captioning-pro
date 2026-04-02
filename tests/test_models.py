import pytest
import tensorflow as tf
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder

def test_encoder_architecture(config):
    # Use weights=None for fast testing of the code structure
    encoder = CNN_Encoder(config)
    sample_img = tf.random.uniform((1, 299, 299, 3))
    output = encoder(sample_img)
    assert len(output.shape) == 3
    assert output.shape[-1] == config['model']['embedding_dim']

def test_decoder_architecture(config):
    decoder = RNN_Decoder(config)
    batch_size = 2
    dummy_input = tf.random.uniform((batch_size, 1))
    dummy_features = tf.random.uniform((batch_size, 64, config['model']['embedding_dim']))
    
    # Test initialization
    hidden = decoder.init_decoder_state(tf.random.uniform((batch_size, config['model']['units'])))
    
    # Test forward pass
    output, state = decoder(dummy_input, hidden)
    assert output.shape == (batch_size, config['dataset']['vocab_size'])