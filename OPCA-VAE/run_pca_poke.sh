#!/bin/bash 
#SBATCH --job-name=vqvae         #  
#SBATCH --output=logs_pca_poke/vqvae_%j.out  #  
#SBATCH --error=logs_pca_poke/vqvae_%j.err 
#SBATCH --nodes=1                #  
#SBATCH --ntasks=1               #  
#SBATCH --cpus-per-task=4        #  
#SBATCH --gres=gpu:1             #  
#SBATCH --time=6-23:59:59        #  
#SBATCH --partition=ciaq         #  
#SBATCH --mem=16G

echo "Job started on $(date)" 
echo "Running on node: $(hostname)"  

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python pca_poke_tools.py \
#   --config configs_rebuild/pca_vae_single_vector_100p.yaml \
#   --ckpt   logs_pca_single_vector/VQVAE_FACE_single_vector_100p/version_0/checkpoints/last.ckpt \
#   basis --comp 0..90 --scale 1.0 \
#         --out_dir output/basis_batch_new \
#         --grid_out output/basis_grid_new_0_9.png \
#         --grid_cols 10

python pca_poke_tools.py \
  --config configs_rebuild/pca_vae_single_vector_100p.yaml \
  --ckpt   logs_pca_single_vector/VQVAE_FACE_single_vector_100p/version_0/checkpoints/last.ckpt \
  edit --image 00000.jpg --comp "8,10" --delta="-2.0..2.0" --step 0.6 \
       --apply_scope all \
       --out_dir output/scan_c9_11_all \
       --scan_grid output/scan_c9_11_strip.png \
       --scan_cols 10
echo "Job finished on $(date)" 