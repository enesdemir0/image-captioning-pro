import os
import json
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
    
    # 1. Professional Naming
    enc_name = config['model']['encoder_name']
    dec_type = config['model']['decoder_type']
    layers = config['model']['num_layers']
    subset = config['dataset']['subset_size']
    target_epochs = config['training']['epochs']
    
    model_id = f"ENC_{enc_name}_DEC_{dec_type}_L{layers}_S{subset}_E{target_epochs}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    # 2. Data Setup
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0:
        img_paths, captions = img_paths[:subset], captions[:subset]

    (tr_i, tr_c), (vl_i, vl_c), (ts_i, ts_c) = loader.split_data(img_paths, captions)
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    # 3. Model Init
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    # Build models to allow loading
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(tf.zeros((1, 1)), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    # --- RESUME LOGIC WITH EPOCH TRACKING ---
    ckpt_dir = config['training']['checkpoint_path']
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
    
    enc_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")
    meta_path = os.path.join(ckpt_dir, f"{model_id}_meta.json")

    start_epoch = 0
    if os.path.exists(enc_path) and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            start_epoch = metadata['last_completed_epoch']
        
        if start_epoch < target_epochs:
            print(f"🔄 Resuming {model_id} from Epoch {start_epoch}. Only {target_epochs - start_epoch} left!")
            encoder.load_weights(enc_path)
            decoder.load_weights(dec_path)
        else:
            print(f"✅ Model already finished {target_epochs} epochs. Skipping training.")
            return

    # 4. Training Loop
    with mlflow.start_run(run_name=f"Training_from_E{start_epoch}"):
        mlflow.log_params(config['model'])
        
        # We start from start_epoch and go until target_epochs
        for epoch in range(start_epoch, target_epochs):
            t_loss = 0
            for batch, (img, target) in enumerate(train_ds):
                t_loss += trainer.train_step(img, target)
            
            v_loss = 0
            for v_batch, (v_img, v_target) in enumerate(val_ds):
                v_loss += trainer.train_step(v_img, v_target)
            
            avg_t, avg_v = (t_loss/(batch+1)).numpy(), (v_loss/(v_batch+1)).numpy()
            mlflow.log_metric("train_loss", avg_t, step=epoch)
            mlflow.log_metric("val_loss", avg_v, step=epoch)
            print(f"Epoch {epoch+1}/{target_epochs} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

            # SAVE EVERY 5 EPOCHS AND UPDATE METADATA
            if (epoch + 1) % 5 == 0 or (epoch + 1) == target_epochs:
                encoder.save_weights(enc_path)
                decoder.save_weights(dec_path)
                with open(meta_path, 'w') as f:
                    json.dump({'last_completed_epoch': epoch + 1}, f)
                print(f"💾 Checkpoint & Metadata saved at Epoch {epoch+1}")

if __name__ == "__main__":
    main()