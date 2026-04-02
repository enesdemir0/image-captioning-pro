import tensorflow as tf
import time
import mlflow

class CaptionTrainer:
    """
    Handles the training loop for Phase 1 (Baseline) and Phase 2 (Teacher Forcing).
    """
    def __init__(self, encoder, decoder, text_processor, config):
        self.encoder = encoder
        self.decoder = decoder
        self.text_processor = text_processor
        self.config = config
        
        # 1. Setup Loss and Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['training']['learning_rate']
        )
        # We use SparseCategoricalCrossentropy because our targets are integers
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )

    def loss_function(self, real, pred):
        """Calculates loss while ignoring the padding tokens."""
        mask = tf.math.logical_not(tf.math.equal(real, 0)) # 0 is <pad>
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        """Processes one batch of images and captions."""
        loss = 0
        batch_size = img_tensor.shape[0]
        
        # Initialize the decoder state using the encoder output
        # (This is the 'Dense Map' connection we built earlier)
        features = self.encoder(img_tensor)
        # For baseline, we use the mean of features to initialize the state
        mean_features = tf.reduce_mean(features, axis=1)
        hidden = self.decoder.init_decoder_state(mean_features)

        # The first input is always the <start> token
        dec_input = tf.expand_dims(
            [self.text_processor.tokenizer.word_index['<start>']] * batch_size, 1
        )

        with tf.GradientTape() as tape:
            # Loop through the sentence (except the last word)
            for i in range(1, target.shape[1]):
                # 1. Predict the next word
                predictions, hidden = self.decoder(dec_input, hidden)
                
                # 2. Calculate loss against the actual word
                loss += self.loss_function(target[:, i], predictions)

                # --- PHASE 1 LOGIC (Greedy/Autoregressive) ---
                if not self.config['training'].get('use_teacher_forcing', False):
                    # Use the model's own prediction as the next input
                    predicted_id = tf.argmax(predictions, axis=1)
                    dec_input = tf.expand_dims(predicted_id, 1)
                
                # --- PHASE 2 LOGIC (Teacher Forcing) ---
                else:
                    # Use the 'Ground Truth' word as the next input
                    dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        return total_loss
