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
        """Bonus #1: Grey Wolf Optimizer (GWO) inspired decay."""
        # a decreases linearly from 2 to 0
        new_lr = self.initial_lr * (1.0 - (epoch / total_epochs))
        self.optimizer.learning_rate.assign(max(new_lr, 1e-6))
        return new_lr

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0
        batch_size = img_tensor.shape[0]
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            hidden = self.decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
            dec_input = tf.expand_dims([self.text_processor.tokenizer.word_index['<start>']] * batch_size, 1)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                if not self.config['training'].get('use_teacher_forcing', False):
                    dec_input = tf.expand_dims(tf.argmax(predictions, axis=1), 1)
                else:
                    dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, trainable_vars), trainable_vars))
        return total_loss