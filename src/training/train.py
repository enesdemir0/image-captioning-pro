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
    
    # 2. Generate a Unique Model Identifier
    model_id = f"{config['model']['encoder_name']}_{config['model']['decoder_type']}_L{config['model']['num_layers']}"
    
    # 3. Initialize DagsHub/MLflow
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # 4. Initialize Data Pipeline
    print(f"--- Initializing Data for {model_id} ---")
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    # Apply subset limit from config
    subset_size = config['dataset'].get('subset_size', 5000)
    img_paths, captions = img_paths[:subset_size], captions[:subset_size]

    # Split into Train (70%), Val (15%), Test (15%)
    (tr_i, tr_c), (vl_i, vl_c), (ts_i, ts_c) = loader.split_data(img_paths, captions)

    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    # 5. Initialize Models
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    # 6. Start Experiment Run
    with mlflow.start_run(run_name=model_id):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("model_id", model_id)

        print(f"Training {model_id} for {config['training']['epochs']} epochs...")
        
        for epoch in range(config['training']['epochs']):
            # --- Training Loop ---
            t_loss = 0
            for batch, (img, target) in enumerate(train_ds):
                t_loss += trainer.train_step(img, target)
            
            avg_t_loss = (t_loss / (batch + 1)).numpy()
            mlflow.log_metric("train_loss", avg_t_loss, step=epoch)

            # --- Validation Loop ---
            v_loss = 0
            for v_batch, (v_img, v_target) in enumerate(val_ds):
                v_loss += trainer.train_step(v_img, v_target) # Gradients not updated in val
            
            avg_v_loss = (v_loss / (v_batch + 1)).numpy()
            mlflow.log_metric("val_loss", avg_v_loss, step=epoch)

            print(f"Epoch {epoch+1} | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}")

        # --- Final Model Saving ---
        ckpt_dir = config['training']['checkpoint_path']
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Unique names: e.g. InceptionV3_GRU_L3_encoder.weights.h5
        enc_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
        dec_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")
        
        encoder.save_weights(enc_path)
        decoder.save_weights(dec_path)
        print(f"✅ Final weights saved: {enc_path}")

if __name__ == "__main__":
    main()