import os
import tensorflow as tf
import mlflow
import dagshub
from src.models import encoder
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.training.trainer import CaptionTrainer

def main():
    # 1. Load Configuration
    config = load_config()
    
    # 2. Initialize DagsHub for Remote MLflow Tracking
    # This will ask for your token or login when you run it in Colab
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
    
    # Subset for initial experiments
    subset_size = config['dataset'].get('subset_size', 5000)
    dataset = loader.get_dataset(img_paths[:subset_size], captions[:subset_size])

    # 4. Initialize Models
    print(f"Building Models: {config['model']['encoder_name']} + {config['model']['decoder_type']}")
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)

    # 5. Initialize the Trainer (The Engine)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    # 6. Start MLflow Run
    run_name = f"{config['model']['encoder_name']}_{config['model']['decoder_type']}_L{config['model']['num_layers']}"
    
    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters from config
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("subset_size", subset_size)

        print(f"Starting Training for {config['training']['epochs']} epochs...")
        epochs = config['training']['epochs']
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Progress bar simulation for terminal
            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss = trainer.train_step(img_tensor, target)
                total_loss += batch_loss

                if batch % 50 == 0:
                    print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

            # Calculate and log epoch loss
            epoch_loss = total_loss / (batch + 1)
            mlflow.log_metric("loss", epoch_loss.numpy(), step=epoch)
            print(f'--- Epoch {epoch+1} Average Loss: {epoch_loss:.4f} ---')

            # Optional: Save a model checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                ckpt_dir = config['training']['checkpoint_path']
                os.makedirs(ckpt_dir, exist_ok=True)
                # Change .h5 to .weights.h5
                encoder.save_weights(os.path.join(ckpt_dir, f"encoder_epoch_{epoch+1}.weights.h5"))
                decoder.save_weights(os.path.join(ckpt_dir, f"decoder_epoch_{epoch+1}.weights.h5"))
                print(f"Saved checkpoint for epoch {epoch+1}")

        print("Training Complete!")
        # Final Log: Save the total number of parameters as a tag
        mlflow.set_tag("status", "completed")

if __name__ == "__main__":
    main()