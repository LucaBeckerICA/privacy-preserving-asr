import sys
import argparse
import yaml
import os
import torch
import logging
from datetime import datetime
import warnings
from custom_utils.util import get_run_name
import numpy as np
import random

# You may want to suppress warnings due to package inconsistencies
warnings.filterwarnings("ignore")
from speaker_verification_hf.trainer_single import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_seeds(seed=0):
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(config):
    """Main function for training speaker verification model."""
    
    # Setup log directories (only on rank 0)
    init_seeds(0)
    log_root_path = config["speaker_verification"]["log_root_path"]
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H:%M"))
    log_dir_trainer = os.path.join(log_dir, 'trainer')
    log_config_path = os.path.join(log_dir, 'config.yaml')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir_trainer, exist_ok=True)

    # Save the config file
    with open(log_config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("BEGIN TRAINING")

    # Initialize trainer
    train_args = config["speaker_verification"]["speaker_verification_trainer"]
    train_args["log_dir"] = log_dir_trainer

    run_name = get_run_name(config)
    run_folder = os.path.join(config['speaker_verification']['log_root_path'], run_name)

    os.makedirs(run_folder, exist_ok=True)

    trainer = Trainer(config=config, run_folder=run_folder, **train_args)
    
    epochs = config["speaker_verification"]["epochs"]
    # Training loop
    
    val_eers = []
    val_min_dcfs = []
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Train model
        train_loss = trainer.train_network(epoch=epoch)
        trainer.save_checkpoint(epoch)
        logger.info(f"Done training epoch {epoch}")
        logger.info(f"Saving model epoch {epoch}")
        

        logger.info(f"Done saving model epoch {epoch}")
        logger.info(f"Saved model {epoch + 1}/{epochs}, Train Loss: {train_loss}")

        # Evaluate model
        eer, min_dcf = trainer.eval_network(mode='val')
        val_eers.append(eer)
        val_min_dcfs.append(min_dcf)

        logger.info(f"Done evaluating model epoch {epoch}")
        logger.info(f"Validation EER: {eer}, MinDCF: {min_dcf}")

    best_epoch = np.argmin(val_eers)
    logger.info(f"Loading best checkpoint from epoch {best_epoch} with validation EER {val_eers[best_epoch]}")
    trainer.load_checkpoint(best_epoch)
    
    # Final evaluation on test set
    eer, min_dcf = trainer.eval_network(mode='test')

    logger.info(f"Test EER: {eer}, MinDCF: {min_dcf}")
    logger.info("RUN SUCCESSFULL")
    
    with open(os.path.join(run_folder, 'spv_metrics.txt'), 'w') as f:
        f.write(f"EER: {eer}, minDCF: {min_dcf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AV speaker recognition on VoxCeleb2")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Name of the config file')
    #parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # Load config file
    config_path = args.config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Run main function
    main(config)
