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
    # 0. Load the Brain (Config)
    config = load_config()
    
    # 1. Pull dynamic values from config (No hardcoding!)
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    subset = config['dataset']['subset_size']
    
    # --- YOUR ORIGINAL METADATA NAMING (DNA STRING) ---
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    attn = config['model'].get('attention_type', 'None')
    opt = "Adam" if config['training'].get('optimizer_type') == "adam" else "GWO"
    tf_val = "TF" if config['training']['use_teacher_forcing'] else "Base"

    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    # 2. Init DagsHub & MLflow Logic (Exactly as you had it)
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted': client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(model_id)

    # 3. Data Pipeline
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (tr_i, tr_c), (vl_i, vl_c), _ = loader.split_data(img_paths, captions)
    
    # Use batch_size from config here
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=batch_size, is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=batch_size, is_training=False)

    # 4. Model Initialization
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)
    
    # Dry run for shapes
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else (49 if enc == "VGG16" else 64)
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, num_f, units))))

    # 5. Checkpoint Logic (Exactly as you had it)
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
            encoder.load_weights(enc_path); decoder.load_weights(dec_path)
        else:
            print("✨ Model already fully trained."); return

    # --- THE CRITICAL DYNAMIC STEPS CALCULATION ---
    # This ensures the loop finishes based on your data size and batch size!
    steps_per_epoch = max(1, len(tr_i) // batch_size)
    val_steps = max(1, len(vl_i) // batch_size)

    # 6. The Training Loop
    with mlflow.start_run(run_name="Training_Session"):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        
        for epoch in range(start_epoch, epochs):
            if config['training'].get('optimizer_type') == "greywolf":
                current_lr = trainer.update_metaheuristic_lr(epoch, epochs)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # --- Training Phase ---
            t_l = 0
            # .take() is required because of .repeat() in the loader
            for batch, (img, tgt) in enumerate(train_ds.take(steps_per_epoch)):
                t_l += trainer.train_step(img, tgt)
                if batch % 2 == 0: # Print every 2 batches for small tests
                    print(f"Epoch {epoch+1} Batch {batch}/{steps_per_epoch} Loss {t_l/(batch+1):.4f}")
            
            # --- Validation Phase (Uses TEST_STEP: No weight updates!) ---
            v_l = 0
            for v_batch, (v_img, v_tgt) in enumerate(val_ds.take(val_steps)):
                v_l += trainer.test_step(v_img, v_tgt)
            
            # Final Metrics for the epoch
            train_loss = (t_l / steps_per_epoch).numpy()
            val_loss = (v_l / val_steps).numpy()
            
            # --- PRINTING TO CONSOLE ---
            print(f"⭐ Epoch {epoch+1} DONE | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # --- LOGGING TO MLFLOW ---
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            # --- SAVE CHECKPOINTS ---
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                encoder.save_weights(enc_path); decoder.save_weights(dec_path)
                with open(meta_path, 'w') as f: json.dump({'last_completed_epoch': epoch+1}, f)
                print(f"💾 Checkpoint saved.")

if __name__ == "__main__":
    main()