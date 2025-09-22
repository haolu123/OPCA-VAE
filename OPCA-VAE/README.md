<h1 align="center">
  <b>PCA-VAE for image Reconstructions</b><br>
</h1>

##### This Repository is adapted from https://github.com/AntixK/PyTorch-VAE. Appreciate it!

**Update 9/22/2025:** 
- Adjusted PyTorch Lightning import commands to support 2.5.2 version
- Adjusted Parrallel Online PCA for pathed laten space (when keep_shape = True)

### Description
This work is trying to use online PCA to replace VQ in VQ-VAE. (VQ is a online k-means algorithm) 
Our current experiment shows that OPCA(online PCA)-VAE  can do the same work as VQ-VAE on reconstruction cases. However, we are not quite sure if it also works are generation and classification task. Also I'm thinking maybe we can test if the latent represtation of OPCA-VAE has the same "interpret ability" as beta-VAE. These need experiments to test.

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 (tested up to 2.5.2)
- All experiments were run on a CUDA-enabled GPU

### dataset
Currently, we used https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256.

### Usage
All usable model codes are saved at .OPCA-VAE/models/pca_vae_v2.py
./experiment.py contains the PyTorch Lightning train, valid, test processes.
do
python run.py -c configs_rebuild/pca_vae.yaml 
to run the code

Notice: in ./models/pca_vae_v2.py, line 205-207 are write for stable update of OPCA. If it converged very slow, please consider to delete it or increase the learning rate of OPCA (times another scaler before lr)

**Config file template - See folder "configs"**

```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 3
  latent_dim: 
    .         # Other parameters required by the model
    .
    .

data_params:
  data_path: "<path to the dataset>"
  train_batch_size: 64 # Better to have a square number
  val_batch_size:  64
  patch_size: 64  # Models are designed to work for this size
  num_workers: 4
  
exp_params:
  manual_seed: 1265
  LR: 0.005
  weight_decay:
    .         # Other arguments required for training, like scheduler etc.
    .
    .

trainer_params:
  gpus: 1         
  max_epochs: 100
  gradient_clip_val: 1.5
    .
    .
    .

logging_params:
  save_dir: "logs/"
  name: "<experiment name>"
```

**View TensorBoard Logs**
```
$ cd logs/<experiment name>/version_<the version you want>
$ tensorboard --logdir .
```


