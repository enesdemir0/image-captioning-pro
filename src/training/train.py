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
    
    # --- METADATA NAMING ---
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    subset, epochs = config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "GWO" if config['training'].get('optimizer_type') == "greywolf" else "Adam"
    tf_val = "TF" if config['training']['use_teacher_forcing'] else "Base"

    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted': client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0: img_paths, captions = img_paths[:subset], captions[:subset]
    (tr_i, tr_c), (vl_i, vl_c), _ = loader.split_data(img_paths, captions)
    train_ds = loader.get_dataset(tr_i, tr_c, batch_size=config['training']['batch_size'], is_training=True)
    val_ds = loader.get_dataset(vl_i, vl_c, batch_size=config['training']['batch_size'], is_training=False)

    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    trainer = CaptionTrainer(encoder, decoder, loader.text_processor, config)
    
    # Build
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, units))))

    ckpt_dir = config['training']['checkpoint_path']
    os.makedirs(ckpt_dir, exist_ok=True)
    enc_path, dec_path = os.path.join(ckpt_dir, f"{model_id}_enc.weights.h5"), os.path.join(ckpt_dir, f"{model_id}_dec.weights.h5")
    meta_path = os.path.join(ckpt_dir, f"{model_id}_meta.json")

    start_epoch = 0
    if os.path.exists(enc_path) and os.path.exists(meta_path):
        with open(meta_path, 'r') as f: start_epoch = json.load(f)['last_completed_epoch']
        if start_epoch < epochs:
            encoder.load_weights(enc_path); decoder.load_weights(dec_path)
        else: return

    with mlflow.start_run(run_name="Training"):
        mlflow.log_params(config['model']); mlflow.log_params(config['training'])
        for epoch in range(start_epoch, epochs):
            if opt == "GWO":
                current_lr = trainer.update_metaheuristic_lr(epoch, epochs)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            t_l = 0
            for batch, (img, tgt) in enumerate(train_ds): t_l += trainer.train_step(img, tgt)
            v_l = 0
            for v_batch, (v_img, v_tgt) in enumerate(val_ds): v_l += trainer.train_step(v_img, v_tgt)
            
            mlflow.log_metrics({"train_loss": (t_l/(batch+1)).numpy(), "val_loss": (v_l/(v_batch+1)).numpy()}, step=epoch)
            print(f"Epoch {epoch+1} | Train: {t_l/(batch+1):.4f} | Val: {v_l/(v_batch+1):.4f}")

            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                encoder.save_weights(enc_path); decoder.save_weights(dec_path)
                with open(meta_path, 'w') as f: json.dump({'last_completed_epoch': epoch+1}, f)

if __name__ == "__main__":
    main()