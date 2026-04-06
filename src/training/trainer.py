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
        loss = 0
        batch_size = img_tensor.shape[0]
        
        with tf.GradientTape() as tape:
            # 1. Get the GRID of features (e.g., 8x8 = 64 spots)
            features = self.encoder(img_tensor)
            
            # 2. Use the MEAN only for the FIRST initialization
            mean_features = tf.reduce_mean(features, axis=1)
            hidden = self.decoder.init_decoder_state(mean_features)

            dec_input = tf.expand_dims(
                [self.text_processor.tokenizer.word_index['<start>']] * batch_size, 1
            )

            for i in range(1, target.shape[1]):
                # PASS THE WHOLE GRID (features) TO THE DECODER
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                
                loss += self.loss_function(target[:, i], predictions)

                if not self.config['training'].get('use_teacher_forcing', False):
                    predicted_id = tf.argmax(predictions, axis=1)
                    dec_input = tf.expand_dims(predicted_id, 1)
                else:
                    dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1])) 
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        return total_loss
