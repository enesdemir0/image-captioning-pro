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
    
    # --- DYNAMIC PROFESSIONAL NAMING ---
    enc = config['model']['encoder_name']
    dec = config['model']['decoder_type']
    layers = config['model']['num_layers']
    subset = config['dataset']['subset_size']
    epochs = config['training']['epochs']
    
    # Phase 2 Label
    tf_label = "TF" if config['training'].get('use_teacher_forcing', False) else "Base"
    
    # Phase 3 Label (Default to 'None' for Phase 1 & 2)
    attn_type = config['model'].get('attention_type', 'None')
    
    # UNIVERSAL ID: e.g., ENC_Xception_DEC_LSTM_L3_S50000_E50_TF_Bahdanau
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_S{subset}_E{epochs}_{tf_label}_{attn_type}"
    
    # 2. Initialize DagsHub/MLflow
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    # RESTORE IF DELETED FIX
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted':
        print(f"♻️ Restoring deleted experiment: {model_id}")
        client.restore_experiment(exp.experiment_id)
    
    mlflow.set_experiment(model_id)

    # 3. Data Setup
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0:
        img_paths, captions = img_paths[:subset], captions[:subset]

    (tr_i, tr_c), (vl_i, vl_c), (ts_i, ts_c) = loader.split_data(img_paths, captions)
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    # 4. Model Init
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)
    
    # Force Build
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(tf.zeros((1, 1)), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    # Checkpoint Pathing (on Drive)
    ckpt_dir = config['training']['checkpoint_path']
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
    enc_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")
    meta_path = os.path.join(ckpt_dir, f"{model_id}_meta.json")

    # Resume Logic
    start_epoch = 0
    if os.path.exists(enc_path) and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            start_epoch = json.load(f)['last_completed_epoch']
        if start_epoch < epochs:
            print(f"🔄 Resuming from Epoch {start_epoch}...")
            encoder.load_weights(enc_path)
            decoder.load_weights(dec_path)
        else:
            print("✅ Experiment already completed.")
            return

    # 5. Training Loop
    with mlflow.start_run(run_name="Training_Session"):
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        
        print(f"🚀 Launching: {model_id}")
        for epoch in range(start_epoch, epochs):
            t_loss = 0
            for batch, (img, target) in enumerate(train_ds):
                t_loss += trainer.train_step(img, target)
            
            v_loss = 0
            for v_batch, (v_img, v_target) in enumerate(val_ds):
                v_loss += trainer.train_step(v_img, v_target)
            
            avg_t, avg_v = (t_loss/(batch+1)).numpy(), (v_loss/(v_batch+1)).numpy()
            mlflow.log_metric("train_loss", avg_t, step=epoch)
            mlflow.log_metric("val_loss", avg_v, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

            # Save every 5 epochs
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                encoder.save_weights(enc_path)
                decoder.save_weights(dec_path)
                with open(meta_path, 'w') as f:
                    json.dump({'last_completed_epoch': epoch + 1}, f)
                print(f"💾 Checkpoint saved at Epoch {epoch+1}")

if __name__ == "__main__":
    main()