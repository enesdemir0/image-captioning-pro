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
    # 1. Load Configuration
    config = load_config()
    
    # 2. GENERATE AUTOMATIC NAME (Matches what you want)
    enc_name = config['model']['encoder_name']
    dec_type = config['model']['decoder_type']
    layers = config['model']['num_layers']
    model_id = f"Encoder_{enc_name}_Decoder_{dec_type}_L{layers}"
    
    # 3. Initialize DagsHub/MLflow
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # --- THIS MATCHES THE EXPERIMENT NAME TO YOUR MODEL ID ---
    mlflow.set_experiment(model_id)

    # 4. Initialize Data Pipeline
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    subset_size = config['dataset'].get('subset_size', 0)
    if subset_size > 0:
        img_paths, captions = img_paths[:subset_size], captions[:subset_size]

    (tr_i, tr_c), (vl_i, vl_c), (ts_i, ts_c) = loader.split_data(img_paths, captions)
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    # 5. Initialize Models
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    # 6. Start Experiment Run (Named "Training")
    with mlflow.start_run(run_name="Training_Phase"):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])

        print(f"Starting Training for Experiment: {model_id}")
        
        for epoch in range(config['training']['epochs']):
            t_loss = 0
            for batch, (img, target) in enumerate(train_ds):
                t_loss += trainer.train_step(img, target)
            
            v_loss = 0
            for v_batch, (v_img, v_target) in enumerate(val_ds):
                v_loss += trainer.train_step(v_img, v_target)
            
            avg_t = (t_loss / (batch + 1)).numpy()
            avg_v = (v_loss / (v_batch + 1)).numpy()
            
            mlflow.log_metric("train_loss", avg_t, step=epoch)
            mlflow.log_metric("val_loss", avg_v, step=epoch)
            print(f"Epoch {epoch+1} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

        # Final Save
        ckpt_dir = config['training']['checkpoint_path']
        os.makedirs(ckpt_dir, exist_ok=True)
        encoder.save_weights(os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5"))
        decoder.save_weights(os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5"))
        print(f"✅ Weights saved for {model_id}")

if __name__ == "__main__":
    main()