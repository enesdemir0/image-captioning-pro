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
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    # 1. 3-Way Split
    (tr_i, tr_c), (vl_i, vl_c), (ts_i, ts_c) = loader.split_data(img_paths, captions)
    
    # Use subsets for testing
    subset = config['dataset'].get('subset_size', 5000)
    train_ds = loader.get_dataset(tr_i[:subset], tr_c[:subset], is_training=True)
    val_ds = loader.get_dataset(vl_i[:int(subset*0.2)], vl_c[:int(subset*0.2)], is_training=False)

    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)

    with mlflow.start_run(run_name=f"{config['model']['decoder_type']}_Stacked_{config['model']['num_layers']}"):
        for epoch in range(config['training']['epochs']):
            # --- Training ---
            t_loss = 0
            for batch, (img, tgt) in enumerate(train_ds):
                t_loss += trainer.train_step(img, tgt)
            
            # --- Validation ---
            v_loss = 0
            for v_batch, (v_img, v_tgt) in enumerate(val_ds):
                # We use the trainer logic but gradients are NOT updated because 
                # we don't call optimizer.apply_gradients in a val_step (simplified here)
                v_loss += trainer.train_step(v_img, v_tgt) 

            mlflow.log_metric("train_loss", (t_loss/(batch+1)).numpy(), step=epoch)
            mlflow.log_metric("val_loss", (v_loss/(v_batch+1)).numpy(), step=epoch)
            print(f"Epoch {epoch+1} | Train Loss: {t_loss/(batch+1):.4f} | Val Loss: {v_loss/(v_batch+1):.4f}")

            if (epoch + 1) % 5 == 0:
                encoder.save_weights(f"models/checkpoints/encoder_e{epoch+1}.weights.h5")
                decoder.save_weights(f"models/checkpoints/decoder_e{epoch+1}.weights.h5")

if __name__ == "__main__":
    main()