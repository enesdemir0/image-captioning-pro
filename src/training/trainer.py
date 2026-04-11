import tensorflow as tf

class CaptionTrainer:
    def __init__(self, encoder, decoder, text_processor, config):
        self.encoder = encoder
        self.decoder = decoder
        self.text_processor = text_processor
        self.config = config
        
        self.initial_lr = config['training']['learning_rate']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_lr)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )

    def update_metaheuristic_lr(self, epoch, total_epochs):
        """Bonus #1: Grey Wolf Optimizer (GWO) inspired linear decay."""
        # In GWO, 'a' decreases from 2 to 0. Here we scale the LR similarly.
        new_lr = self.initial_lr * (1.0 - (epoch / total_epochs))
        self.optimizer.learning_rate.assign(max(new_lr, 1e-6))
        return new_lr

    def loss_function(self, real, pred):
        # Mask the loss so we don't penalize the <pad> tokens (0)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0
        batch_size = img_tensor.shape[0]
        
        # Initialize the hidden state (using our new mean-feature logic)
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            hidden = self.decoder.init_decoder_state(features)
            
            # Start token for the whole batch
            dec_input = tf.expand_dims([self.text_processor.tokenizer.word_index['<start>']] * batch_size, 1)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                
                # Teacher Forcing logic
                if self.config['training'].get('use_teacher_forcing', False):
                    dec_input = tf.expand_dims(target[:, i], 1)
                else:
                    dec_input = tf.expand_dims(tf.argmax(predictions, axis=1), 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return total_loss

    @tf.function
    def test_step(self, img_tensor, target):
        """Validation step: NO gradients, NO optimizer."""
        loss = 0
        batch_size = img_tensor.shape[0]
        
        features = self.encoder(img_tensor)
        hidden = self.decoder.init_decoder_state(features)
        dec_input = tf.expand_dims([self.text_processor.tokenizer.word_index['<start>']] * batch_size, 1)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            loss += self.loss_function(target[:, i], predictions)
            
            # During validation, we NEVER use teacher forcing (to see real performance)
            dec_input = tf.expand_dims(tf.argmax(predictions, axis=1), 1)

        return (loss / int(target.shape[1]))