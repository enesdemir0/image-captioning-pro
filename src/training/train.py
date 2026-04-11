import os
import json
import tensorflow as tf
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.training.trainer import CaptionTrainer

def main():
    config = load_config()
    
    # --- METADATA NAMING (The Engine ID) ---
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    subset, epochs = config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "GWO" if config['training'].get('optimizer_type') == "greywolf" else "Adam"
    tf_val = "TF" if config['training']['use_teacher_forcing'] else "Base"

    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    # Init DagsHub & MLflow
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Restore deleted experiments if needed
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted': client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(model_id)

    # 1. Data Pipeline
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (tr_i, tr_c), (vl_i, vl_c), _ = loader.split_data(img_paths, captions)
    
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    # 2. Model Initialization
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)
    
    # 3. Build Models (Dry run for shapes)
    encoder(tf.zeros((1, 299, 299, 3)))
    # Xception 10x10=100, VGG 7x7=49
    num_f = 100 if enc == "Xception" else (49 if enc == "VGG16" else 64)
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, num_f, units))))

    # 4. Checkpoint Logic
    ckpt_dir = config['training']['checkpoint_path']
    os.makedirs(ckpt_dir, exist_ok=True)
    enc_path = os.path.join(ckpt_dir, f"{model_id}_enc.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_dec.weights.h5")
    meta_path = os.path.join(ckpt_dir, f"{model_id}_meta.json")

    start_epoch = 0
    if os.path.exists(enc_path) and os.path.exists(meta_path):
        with open(meta_path, 'r') as f: start_epoch = json.load(f)['last_completed_epoch']
        if start_epoch < epochs:
            print(f"🔄 Resuming from Epoch {start_epoch}...")
            encoder.load_weights(enc_path)
            decoder.load_weights(dec_path)
        else:
            print("✨ Model already fully trained.")
            return

    # 5. The Training Loop
    with mlflow.start_run(run_name="Training_Session"):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        
        for epoch in range(start_epoch, epochs):
            # Apply GWO Learning Rate Decay if selected
            if config['training'].get('optimizer_type') == "greywolf":
                current_lr = trainer.update_metaheuristic_lr(epoch, epochs)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # --- Training Phase ---
            t_l = 0
            for batch, (img, tgt) in enumerate(train_ds):
                t_l += trainer.train_step(img, tgt)
                if batch % 100 == 0: print(f"Epoch {epoch+1} Batch {batch} Loss {t_l/(batch+1):.4f}")
            
            # --- Validation Phase (Uses TEST_STEP: No weight updates!) ---
            v_l = 0
            for v_batch, (v_img, v_tgt) in enumerate(val_ds):
                v_l += trainer.test_step(v_img, v_tgt)
            
            # Logging
            train_loss = (t_l / (batch + 1)).numpy()
            val_loss = (v_l / (v_batch + 1)).numpy()
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"⭐ Epoch {epoch+1} COMPLETED | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save Checkpoints
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                encoder.save_weights(enc_path)
                decoder.save_weights(dec_path)
                with open(meta_path, 'w') as f: 
                    json.dump({'last_completed_epoch': epoch+1}, f)
                print(f"💾 Checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    main()