#%%
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy 
from dataset import VAEDataset
import sys
from utils import instantiate_from_config
from omegaconf import OmegaConf, DictConfig
# torch.use_deterministic_algorithms(True)
#%%
class VerboseEarlyStopping(EarlyStopping): 
    def _run_early_stopping_check(self, trainer): 
        logs = trainer.callback_metrics 
        current = logs.get(self.monitor) 

        if current is None: 
            return  # metric not yet available 
 
        current = current.item() if isinstance(current, torch.Tensor) else float(current) 
 
        if self.best_score is None: 
            self.best_score = current 
            self.wait_count = 0 
            print( 
                f"[EarlyStopping] Epoch {trainer.current_epoch}: Metric {self.monitor} initialized to {current:.4f}", 
                file=sys.stderr 
            ) 
        else: 
            # if self.mode == "min": 
            #     improved = current < self.best_score - self.min_delta 
            # else: 
            improved = current > self.best_score + self.min_delta              # using psnr/ssim, max
    
            if improved: 
                improvement = abs(current - self.best_score) 
                self.best_score = current 
                self.wait_count = 0 
                print(f"[EarlyStopping] Epoch {trainer.current_epoch}: Metric {self.monitor} improved by {improvement:.4f} >= {self.min_delta}. New best: {current:.4f}", 
                    file=sys.stderr) 
            else: 
                self.wait_count += 1 
                if self.wait_count >= self.patience: 
                    trainer.should_stop = True 
                    print( 
                        f"[EarlyStopping] Patience {self.patience} reached. Stopping training at epoch {trainer.current_epoch}.", 
                        file=sys.stderr) 
 

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()                      # choose config file (vq_vae.yaml)
with open(args.filename, 'r') as file:
    try:
        config = OmegaConf.load(file)
        OmegaConf.resolve(config)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

# For reproducibility - obtain seed, model, exp, data, etc info from config file, then train
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])        # exp parameter specific to this "model"

pin_memory = config['trainer_params'].get('accelerator', 'cpu') != 'cpu' 
data = VAEDataset( **config["data_params"], pin_memory=pin_memory ) 

data.setup()

# early stop
early_stop_callback = VerboseEarlyStopping(
    # monitor='val_psnr',                       # early stop monitor visual performance
    monitor='val_ms_ssim',
    patience=20,
    mode='max',
    min_delta=0.002,                          # increase min metric increase
    verbose=False
)

# check point
checkpoint_callback = ModelCheckpoint(
    save_top_k=2, 
    dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
    monitor= "val_psnr",
    save_last= True
)

runner = Trainer(logger=tb_logger,
                 callbacks=[
                    LearningRateMonitor(),
                    checkpoint_callback,
                    early_stop_callback
                 ],
                #  strategy=DDPStrategy(find_unused_parameters=False),
                strategy="auto",
                 log_every_n_steps = 1,                                   # 
                 **config['trainer_params'])


# Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)

runner.test(model=experiment, ckpt_path="best", datamodule=data)

# best_ckpt_path = checkpoint_callback.best_model_path  # safer if multiple callbacks
# use same experiment instance; load best weights into it
# state = torch.load(best_ckpt_path, map_location="cpu")
# experiment.load_state_dict(state["state_dict"], strict=False)
# runner.test(model=experiment, datamodule=data)

# print(f"\n[Testing best checkpoint from]: {best_ckpt_path}")
# if os.path.isfile(best_ckpt_path):
#     runner.test(ckpt_path=best_ckpt_path, datamodule=data)  # Lightning restores the module
# else:
#     print("[Warning] No best checkpoint found. Skipping test evaluation.")