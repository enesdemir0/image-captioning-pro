import os
import tensorflow as tf
import mlflow
import dagshub
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.training.trainer import CaptionTrainer

def main():
    config = load_config()
    enc_name = config['model']['encoder_name']
    dec_type = config['model']['decoder_type']
    layers = config['model']['num_layers']
    model_id = f"Encoder_{enc_name}_Decoder_{dec_type}_L{layers}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    subset_size = config['dataset'].get('subset_size', 50000)
    img_paths, captions = img_paths[:subset_size], captions[:subset_size]

    (tr_i, tr_c), (vl_i, vl_c), (ts_i, ts_c) = loader.split_data(img_paths, captions)
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    # Building models to allow weight loading/saving
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(tf.zeros((1, 1)), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    # --- RESUME LOGIC (Checks Drive folder) ---
    ckpt_dir = config['training']['checkpoint_path']
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
    
    enc_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")

    if os.path.exists(enc_path):
        print(f"🔄 Resuming {model_id} from existing Drive checkpoint...")
        encoder.load_weights(enc_path)
        decoder.load_weights(dec_path)

    with mlflow.start_run(run_name="Training_Phase"):
        mlflow.log_params(config['model'])
        
        for epoch in range(config['training']['epochs']):
            t_loss = 0
            for batch, (img, target) in enumerate(train_ds):
                t_loss += trainer.train_step(img, target)
            
            v_loss = 0
            for v_batch, (v_img, v_target) in enumerate(val_ds):
                v_loss += trainer.train_step(v_img, v_target)
            
            avg_t, avg_v = (t_loss/(batch+1)).numpy(), (v_loss/(v_batch+1)).numpy()
            mlflow.log_metric("train_loss", avg_t, step=epoch)
            mlflow.log_metric("val_loss", avg_v, step=epoch)
            print(f"Epoch {epoch+1} | Train: {avg_t:.4f} | Val: {avg_v:.4f}")

            # --- SAVE EVERY 5 EPOCHS DIRECTLY TO DRIVE ---
            if (epoch + 1) % 5 == 0:
                encoder.save_weights(enc_path)
                decoder.save_weights(dec_path)
                print(f"💾 Checkpoint saved to Drive at Epoch {epoch+1}")

        # Final Save
        encoder.save_weights(enc_path)
        decoder.save_weights(dec_path)
        print("✅ Training Finished. Final weights on Drive.")

if __name__ == "__main__":
    main()