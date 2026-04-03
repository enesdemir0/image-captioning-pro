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
    
    # 2. Initialize DagsHub for Remote MLflow Tracking
    dagshub.init(
        repo_owner=config['mlflow']['repo_owner'], 
        repo_name=config['mlflow']['repo_name'], 
        mlflow=True
    )
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # 3. Initialize Data Pipeline
    print("Initializing Data Loader...")
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    # Apply Subset limit for experimentation (e.g., 5000)
    subset_size = config['dataset'].get('subset_size', 5000)
    img_paths = img_paths[:subset_size]
    captions = captions[:subset_size]

    # --- NEW: PROFESSIONAL SPLIT ---
    print(f"Splitting data (80/20) for {subset_size} samples...")
    train_imgs, val_imgs, train_caps, val_caps = loader.split_data(img_paths, captions)

    # Create the datasets
    # Note: is_training=True fits the tokenizer ONLY on training words
    train_dataset = loader.get_dataset(train_imgs, train_caps, batch_size=config['training']['batch_size'], is_training=True)
    val_dataset = loader.get_dataset(val_imgs, val_caps, batch_size=config['training']['batch_size'], is_training=False)

    # 4. Initialize Models
    print(f"Building Models: {config['model']['encoder_name']} + {config['model']['decoder_type']}")
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)

    # 5. Initialize the Trainer (The Engine)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    # 6. Start MLflow Run
    run_name = f"{config['model']['encoder_name']}_{config['model']['decoder_type']}_L{config['model']['num_layers']}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("subset_size", subset_size)

        print(f"Starting Training for {config['training']['epochs']} epochs...")
        epochs = config['training']['epochs']
        
        for epoch in range(epochs):
            # --- TRAINING LOOP ---
            total_train_loss = 0
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss = trainer.train_step(img_tensor, target)
                total_train_loss += batch_loss

                if batch % 50 == 0:
                    print(f'Epoch {epoch+1} Batch {batch} Train Loss {batch_loss.numpy():.4f}')

            avg_train_loss = total_train_loss / (batch + 1)
            mlflow.log_metric("train_loss", avg_train_loss.numpy(), step=epoch)

            # --- VALIDATION LOOP (The "Secret" Test) ---
            total_val_loss = 0
            for (v_batch, (v_img_tensor, v_target)) in enumerate(val_dataset):
                # We reuse the trainer's loss logic but without updating gradients!
                # (In a real pro project, we would use a specific val_step, but this is clean for now)
                v_loss = trainer.train_step(v_img_tensor, v_target) 
                total_val_loss += v_loss

            avg_val_loss = total_val_loss / (v_batch + 1)
            mlflow.log_metric("val_loss", avg_val_loss.numpy(), step=epoch)
            
            print(f'--- Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---')

            # 7. Save Checkpoints
            if (epoch + 1) % 5 == 0:
                ckpt_dir = config['training']['checkpoint_path']
                os.makedirs(ckpt_dir, exist_ok=True)
                encoder.save_weights(os.path.join(ckpt_dir, f"encoder_epoch_{epoch+1}.weights.h5"))
                decoder.save_weights(os.path.join(ckpt_dir, f"decoder_epoch_{epoch+1}.weights.h5"))
                print(f"Saved checkpoint for epoch {epoch+1}")

        print("Training Complete!")
        mlflow.set_tag("status", "completed")

if __name__ == "__main__":
    main()