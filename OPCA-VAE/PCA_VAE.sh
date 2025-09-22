#!/bin/bash 
#SBATCH --job-name=vqvae         #  
#SBATCH --output=logs_pca/vqvae_%j.out  #  
#SBATCH --error=logs_pca/vqvae_%j.err 
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
python run.py -c configs_rebuild/pca_vae.yaml 

echo "Job finished on $(date)" 