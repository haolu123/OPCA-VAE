import os
import math
import torch
from torch import optim 
from models import BaseVAE 
from models.types_ import * 
import pytorch_lightning as pl 
from torchvision import utils as vutils 
import matplotlib.pyplot as plt 

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as msssim_fn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import lpips
import numpy as np 

class VAEXperiment(pl.LightningModule): 
    def __init__(self, vae_model: BaseVAE, params: dict) -> None: 
        super(VAEXperiment, self).__init__() 

        self.model = vae_model 
        self.params = params 
        self.curr_device = None
        self.psnr = PeakSignalNoiseRatio() 
        self.ssim = StructuralSimilarityIndexMeasure() 
        self.mse_loss_fn = torch.nn.MSELoss()
        self.fid_metric = FrechetInceptionDistance(feature = 2048, reset_real_features = False)         # fid
        self.lpips_metrics = lpips.LPIPS(net='alex').to(self.curr_device if self.curr_device else 'cuda')       # lpips

        # self.register_buffer("codebook_usage", torch.zeros(self.model.num_embeddings, dtype=torch.long))
        # self.codebook_usage_list = []
        self.training_step_outputs = [] 
        self.validation_step_outputs = [] 

        self.train_losses = [] 
        self.val_losses = [] 
    
        # self.usage_ratio_list = []

        self.train_psnrs = []              # train psnr
        self.train_ssims = []              # train ssim
        self.train_mses = []             # MSE
        self.train_ms_ssims = []                           # ms-ssim
        self.train_nlpds = []

        self.train_psnrs_cross_epochs = []
        self.train_ssims_cross_epochs = []              # train ssim
        self.train_mses_cross_epochs = []             # MSE
        self.train_ms_ssims_cross_epochs = []                           # ms-ssim
        self.train_nlpds_cross_epochs = []

        self.val_psnrs = []              # validation psnr
        self.val_ssims = []              # validation ssim
        self.val_mses = []
        self.val_ms_ssims = []
        self.val_fids = []
        self.val_nlpds = []                              # nlpd
        self.val_lpips = []

        self.val_psnrs_list = []              # validation psnr
        self.val_ssims_list = []              # validation ssim
        self.val_mses_list = []
        self.val_ms_ssims_list = []
        self.val_fids_list = []
        self.val_nlpds_list = []                              # nlpd
        self.val_lpips_list = []

        self.test_psnrs = []              # validation psnr
        self.test_ssims = []              # validation ssim
        self.test_mses = []
        self.test_ms_ssims = []
        self.test_fids = []
        self.test_nlpds = []                              # nlpd
        self.test_lpips = []
        self.test_step_outputs = []

    def _get_current_lr(self):
        try:
            # first optimizer, first param group (common case)
            return self.trainer.optimizers[0].param_groups[0]['lr']
        except Exception:
            # during sanity check or if optimizers not yet attached
            return None

    def forward(self, input: Tensor, status='Train', **kwargs) -> Tensor: 
        current_lr = self._get_current_lr() if status == 'Train' else None
        return self.model(input, status=status, current_lr=current_lr) 

    def training_step(self, batch, batch_idx):                                # 
        real_img, labels = batch 
        self.curr_device = real_img.device 

        # Current epoch
        # current_epoch = self.trainer.current_epoch if self.trainer.current_epoch is not None else 0  # starts at 0
        
        # # Total number of epochs (from Trainer's max_epochs)
        # max_epoch = self.trainer.max_epochs 
        recons, input_img, vq_loss, encoding_inds = self.forward(
                real_img,
                status='Train'
            )  
        flat_inds = encoding_inds.flatten()  
        # self.codebook_usage.scatter_add_(0, flat_inds, torch.ones_like(flat_inds, dtype=self.codebook_usage.dtype)) 
        train_loss = self.model.loss_function(recons, input_img, vq_loss, M_N=self.params['kld_weight'], batch_idx=batch_idx) 

        self.training_step_outputs.append(train_loss['loss']) 
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True) 

        sigma2_tensor = torch.tensor(1.0, device=self.curr_device)
        const_term = 0.5 * torch.log(2 * math.pi * sigma2_tensor)

        self.train_psnrs.append(self.psnr(recons, real_img))              # train psnr
        self.train_ssims.append(self.ssim(recons, real_img))              # train ssim
        self.train_mses.append(self.mse_loss_fn(recons, real_img))             # MSE
        self.train_ms_ssims.append(msssim_fn(
                preds=recons,
                target=real_img, 
                kernel_size=3,
                betas=(0.4,0.4,0.2),
                data_range=1.0
            ))                         # ms-ssim

        pixel_mse = (real_img-recons)**2                          # nlpd
        nlpd_map = const_term+pixel_mse/(2*sigma2_tensor)
        nlpd_sample = nlpd_map.mean()
        self.train_nlpds.append(nlpd_sample.cpu())
        return train_loss['loss']

    def on_train_epoch_end(self):                                                # 
        avg_loss = torch.stack(self.training_step_outputs).mean() 
        self.log("train_epoch_loss", avg_loss, prog_bar=True, sync_dist=True) 
        print(f"\n[Epoch {self.current_epoch}] Train Loss: {avg_loss:.4f}") 
        self.train_losses.append(avg_loss.item()) 
        self.training_step_outputs.clear()
        # self.codebook_usage_list.append(self.codebook_usage.cpu().numpy())

        # num_embeddings = self.codebook_usage.numel()
        # num_unused = (self.codebook_usage == 0).sum().item()
        # usage_ratio = 1 - num_unused / num_embeddings
        # self.usage_ratio_list.append(usage_ratio)
        
        # self.codebook_usage.zero_()
 
        # avg metric values of each batch
        avg_train_psnr = torch.stack(self.train_psnrs).mean() 
        avg_train_ssim = torch.stack(self.train_ssims).mean() 
        avg_train_mse = torch.stack(self.train_mses).mean()
        avg_train_ms_ssim = torch.stack(self.train_ms_ssims).mean()
        avg_train_nlpd = torch.stack(self.train_nlpds).mean()
        
        self.train_psnrs.clear()
        self.train_ssims.clear()
        self.train_mses.clear()
        self.train_ms_ssims.clear()
        self.train_nlpds.clear()

        self.log("train_psnr", avg_train_psnr, prog_bar=True, sync_dist=True)  
        self.log("train_ssim", avg_train_ssim, prog_bar=True, sync_dist=True)
        self.log("train_mse", avg_train_mse, prog_bar=True)
        self.log("train_ms_ssim", avg_train_ms_ssim, prog_bar=True, sync_dist=True)
        self.log("train_nlpd", avg_train_nlpd, prog_bar=True, sync_dist=True)
        # self.log("usage_ratio", usage_ratio, prog_bar=True, sync_dist=True)

        # print metrics
        print(
            f"\n[Epoch {self.current_epoch}]"
            f"\nTrain PSNR: {avg_train_psnr:.2f},"
            f"\nTrain SSIM: {avg_train_ssim:.4f},"
            f"\nTrain MSE: {avg_train_mse:.6f}, "
            f"\nTrain MS-SSIM: {avg_train_ms_ssim}, "
            f"\nTrain NLPD: {avg_train_nlpd}"
            # f"\nCodebook usage: used {num_embeddings - num_unused}/{num_embeddings}({usage_ratio*100:.2f}% used)")  
        )
        # append metrics to list
        self.train_psnrs_cross_epochs.append(avg_train_psnr.item()) 
        self.train_ssims_cross_epochs.append(avg_train_ssim.item()) 
        self.train_mses_cross_epochs.append(avg_train_mse.item())
        self.train_ms_ssims_cross_epochs.append(avg_train_ms_ssim.item())
        self.train_nlpds_cross_epochs.append(avg_train_nlpd.item())
 
    def validation_step(self, batch, batch_idx): 
        real_img, labels = batch 
        self.curr_device = real_img.device 

        # Current epoch
        # current_epoch = self.trainer.max_epochs  
        
        # # Total number of epochs (from Trainer's max_epochs)
        # max_epoch = self.trainer.max_epochs

        results = self.forward(
            real_img,
            status='Valid'
            )   
        
        val_loss = self.model.loss_function(*results, M_N=1.0, batch_idx=batch_idx) 
 
        self.validation_step_outputs.append(val_loss['loss']) 
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True) 
 
        recon = results[0]
        self.val_psnrs.append(self.psnr(recon, real_img)) 
        self.val_ssims.append(self.ssim(recon, real_img)) 
        self.val_mses.append(self.mse_loss_fn(recon, real_img))
        self.val_ms_ssims.append(msssim_fn(
            preds=recon, 
            target=real_img, 
            kernel_size=3, 
            betas=(0.4,0.4,0.2),
            data_range=1.0
        ))

        real_255 = (real_img * 255).clamp(0, 255).to(torch.uint8).to(self.curr_device)          # fid
        recon_255 = (recon * 255).clamp(0, 255).to(torch.uint8).to(self.curr_device)
        self.fid_metric.update(real_255, real=True)
        self.fid_metric.update(recon_255, real=False)

        sigma2_tensor = torch.tensor(1.0, device=self.curr_device)
        const_term = 0.5 * torch.log(2 * math.pi * sigma2_tensor)

        pixel_mse = (real_img-recon)**2                                                        # nlpd
        nlpd_map = const_term + pixel_mse/(2*sigma2_tensor)
        nlpd_sample = nlpd_map.mean()
        self.val_nlpds.append(nlpd_sample.cpu())

        with torch.no_grad():
            lpips_value = self.lpips_metrics(recon, real_img).mean()
        self.val_lpips.append(lpips_value)
        return val_loss['loss'] 

    def on_validation_epoch_end(self):                                                # 
        avg_loss = torch.stack(self.validation_step_outputs).mean() 
        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True) 
        print(f"\n[Epoch {self.current_epoch}] Val Loss: {avg_loss:.4f}") 
        self.val_losses.append(avg_loss.item()) 
        self.validation_step_outputs.clear() 

        # average
        avg_psnr = torch.stack(self.val_psnrs).mean() 
        avg_ssim = torch.stack(self.val_ssims).mean()
        avg_mse = torch.stack(self.val_mses).mean()
        fid_score = self.fid_metric.compute()
        avg_ms_ssim = torch.stack(self.val_ms_ssims).mean()
        avg_nlpd = torch.stack(self.val_nlpds).mean()
        avg_lpips = torch.stack(self.val_lpips).mean()

        # append avg metrics to list
        self.val_psnrs_list.append(avg_psnr.item()) 
        self.val_ssims_list.append(avg_ssim.item()) 
        self.val_mses_list.append(avg_mse.item())
        self.val_ms_ssims_list.append(avg_ms_ssim.item())

        self.fid_metric.reset()                         # fid append to list
        self.val_fids_list.append(fid_score.item())

        self.val_nlpds_list.append(avg_nlpd.item())
        self.val_lpips_list.append(avg_lpips.item())
    
        self.val_psnrs.clear()
        self.val_ssims.clear()
        self.val_mses.clear()
        self.val_ms_ssims.clear()
        self.val_nlpds.clear()
        self.val_lpips.clear()
        
        # log
        self.log("val_psnr", avg_psnr, prog_bar=True, sync_dist=True) 
        self.log("val_ssim", avg_ssim, prog_bar=True, sync_dist=True) 
        self.log("val_mse", avg_mse, prog_bar=True)
        self.log("val_fid", fid_score, prog_bar=True, sync_dist=True)
        self.log("val_ms_ssim", avg_ms_ssim, prog_bar=True, sync_dist=True)
        self.log("val_nlpd", avg_nlpd, prog_bar=True, sync_dist=True)
        self.log("val_lpips", avg_lpips, prog_bar=True, sync_dist=True)

        print(
            f"\n[Epoch {self.current_epoch}]"
            f"\nVal PSNR: {avg_psnr:.2f},"
            f"\nVal SSIM: {avg_ssim:.4f},"
            f"\nVal MSE: {avg_mse:.6f},"
            f"\nVal FID: {fid_score:.2f},"
            f"\nVal MS-SSIM: {avg_ms_ssim:.4f},"
            f"\nVal NLPD: {avg_nlpd:.4f},"
            f"\nVal LPIPS: {avg_lpips:.4f}") 
    

    def on_validation_end(self) -> None: 
        self.sample_images() 
 
    def sample_images(self): 
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader())) 
        test_input = test_input.to(self.curr_device) 
        test_label = test_label.to(self.curr_device) 
 
        recons = self.model.generate(test_input, labels=test_label)                      # image reconstruction
        vutils.save_image(recons.data, 
                        os.path.join(self.logger.log_dir, 
                                    "Reconstructions", 
                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"), 
                        normalize=True, 
                        nrow=8) 

    def configure_optimizers(self): 
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay']) 
    
        gamma = self.params.get('scheduler_gamma', 0.0) 
        if gamma > 0.0: 
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) 
            return [optimizer], [scheduler] 
    
        return optimizer
    
    
    @rank_zero_only 
    def on_train_end(self): 

        # training validation loss plot 
        plt.figure() 
        plt.plot(self.train_losses, label="Train Loss") 
        plt.plot(self.val_losses, label="Val Loss") 
        plt.xlabel("Epoch") 
        plt.ylabel("Loss") 
        plt.legend() 
        plt.title("Training and Validation Loss") 
        plt.savefig(os.path.join(self.logger.log_dir, "loss_curve.png")) 
        plt.close() 

        fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 

        metrics = [ 
            ("PSNR", self.train_psnrs_cross_epochs, self.val_psnrs_list), 
            ("SSIM", self.train_ssims_cross_epochs, self.val_ssims_list), 
            ("MS-SSIM", self.train_ms_ssims_cross_epochs, self.val_ms_ssims_list)
        ] 
        for ax, (metric_name, train_values, val_values) in zip(axs, metrics): 
            ax.plot(train_values, label=f"Train {metric_name}", linestyle='-') 
            ax.plot(val_values, label=f"Val {metric_name}", linestyle='--') 
            ax.set_xlabel("Epoch") 
            ax.set_ylabel(metric_name) 
            ax.set_title(f"{metric_name} Metrics") 
            ax.legend() 
            ax.grid(True) 
        plt.tight_layout() 
        plt.savefig(os.path.join(self.logger.log_dir, "psnr_ssim_msssim_curves.png")) 
        plt.close() 

        # mse, nlpd
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

        axs[0].plot(self.train_mses_cross_epochs, label="Train MSE") 
        axs[0].plot(self.val_mses_list, label="Validation MSE") 
        axs[0].axhline(0, color='grey', linestyle='--', label='Reference: MSE=0') 
        axs[0].set_xlabel("Epoch") 
        axs[0].set_ylabel("MSE") 
        axs[0].set_title("Pixel Level MSE") 
        axs[0].legend()
        axs[0].grid(True) 

        axs[1].plot(self.train_nlpds_cross_epochs, label="Train NLPD")
        axs[1].plot(self.val_nlpds_list, label="Val NLPD") 
        axs[1].set_xlabel("Epoch") 
        axs[1].set_ylabel("NLPD") 
        axs[1].set_title("Validation NLPD") 
        axs[1].legend()
        axs[1].grid(True) 

        plt.tight_layout() 
        plt.savefig(os.path.join(self.logger.log_dir, "mse_nlpd_curve.png")) 
        plt.close() 

        # fid, lpips
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

        axs[0].plot(self.val_fids_list, label="Val FID") 
        axs[0].set_xlabel("Epoch") 
        axs[0].set_ylabel("FID") 
        axs[0].set_title("Validation FID") 
        axs[0].legend() 
        axs[0].grid(True) 

        axs[1].plot(self.val_lpips_list, label="Val LPIPS") 
        axs[1].set_xlabel("Epoch") 
        axs[1].set_ylabel("LPIPS") 
        axs[1].set_title("Validation LPIPS") 
        axs[1].legend() 
        axs[1].grid(True) 

        plt.tight_layout() 
        plt.savefig(os.path.join(self.logger.log_dir, "fid_lpips_curve.png")) 
        plt.close()

        # plot codebook usage
        # usage_np = self.codebook_usage_list[-1]
        # sorted_usage = np.sort(usage_np)
        # cdf = np.cumsum(sorted_usage) / np.sum(sorted_usage) 

        # fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

        # # axs[0].plot(range(1, len(cdf) + 1), cdf) 
        # axs[0].set_xlabel("Codebook Entry (sorted)") 
        # axs[0].set_ylabel("CDF") 
        # axs[0].set_title("Codebook Usage CDF") 
        # axs[0].grid(True) 

        # axs[1].bar(range(len(usage_np)), usage_np)
        # axs[1].set_xlabel("Codebook Entry Index")
        # axs[1].set_ylabel("Usage Count")
        # axs[1].set_title("Codebook Usage Histogram")
        # axs[1].grid(True)

        # plt.tight_layout() 
        # plt.savefig(os.path.join(self.logger.log_dir, "codebook_usage_cdf.png"))
        # plt.close() 

        # fig, axs = plt.subplots(1, 1, figsize=(5, 5)) 

        # axs.plot(self.usage_ratio_list) 
        # axs.set_xlabel("epoch") 
        # axs.set_ylabel("codebook usage ratio") 
        # axs.set_title("Codebook Usage ratio vs epochs") 
        # axs.grid(True) 

        # plt.tight_layout() 
        # plt.savefig(os.path.join(self.logger.log_dir, "codebook_usage_ratio.png"))
        # plt.close() 

        metrics_path = os.path.join(self.logger.log_dir, "final_metrics.txt") 
        with open(metrics_path, "w") as f: 
            # Save alpha and beta if available
            try:
                alpha = self.model.alpha
                beta = self.model.beta
                soft_sample = self.model.vq_layer.soft_sample_flag
                q_flag = self.model.vq_layer.q_flag
                mem_flag = self.model.vq_layer.mem_flag
                random_flag = self.model.vq_layer.random_flag
                random_decay_method = self.model.vq_layer.decay_method          # New

                if not soft_sample and not q_flag and not mem_flag and not random_flag:
                    f.write("Traditional VQ-VAE method: \n")
                else:
                    f.write("our method: \n")
                f.write(f"{'Alpha':<15} {'-':>12} {alpha:>12.4f}\n")
                f.write(f"{'Beta':<15} {'-':>12} {beta:>12.4f}\n")
                f.write(f"{'q_flag':<15} {'-':>12} {str(q_flag):>12}\n")  # keep as float
                f.write(f"{'soft_sample':<15} {'-':>12} {str(soft_sample):>12}\n")
                f.write(f"{'mem_flag':<15} {'-':>12} {str(mem_flag):>12}\n")
                f.write(f"{'random_flag':<15} {'-':>12} {str(random_flag):>12}\n")
                f.write(f"{'random_decay_method':<15} {'-':>12} {str(random_decay_method):>12}\n")
            except AttributeError:
                print("[Warning] Alpha and Beta attributes not found in model.")
            f.write("[Final Epoch Metrics]\n") 
            f.write(f"{'Metric':<15} {'Train':>12} {'Validation':>12}\n") 
            f.write("-" * 41 + "\n") 
            f.write(f"{'Loss':<15} {self.train_losses[-1]:>12.6f} {self.val_losses[-1]:>12.6f}\n") 
            f.write(f"{'PSNR':<15} {self.train_psnrs_cross_epochs[-1]:>12.2f} {self.val_psnrs_list[-1]:>12.2f}\n") 
            f.write(f"{'SSIM':<15} {self.train_ssims_cross_epochs[-1]:>12.4f} {self.val_ssims_list[-1]:>12.4f}\n") 
            f.write(f"{'MSE':<15} {self.train_mses_cross_epochs[-1]:>12.6f} {self.val_mses_list[-1]:>12.6f}\n") 
            f.write(f"{'FID':<15} {'-':>12} {self.val_fids_list[-1]:>12.4f}\n") 
            f.write(f"{'MS-SSIM':<15} {self.train_ms_ssims_cross_epochs[-1]:>12.4f} {self.val_ms_ssims_list[-1]:>12.4f}\n") 
            f.write(f"{'NLPD':<15} {self.train_nlpds_cross_epochs[-1]:12.4f} {self.val_nlpds_list[-1]:>12.4f}\n") 
            f.write(f"{'LPIPS':<15} {'-':>12} {self.val_lpips_list[-1]:>12.4f}\n") 

        # # Load best checkpoint and test
        # best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        # print(f"\n[Testing best checkpoint from]: {best_ckpt_path}")
        # if os.path.isfile(best_ckpt_path):
        #     best_model =  self.__class__.load_from_checkpoint(
        #         best_ckpt_path, 
        #         vae_model=self.model, 
        #         params=self.params
        #     )
        #     trainer = self.trainer
        #     trainer.test(model=best_model, datamodule=trainer.datamodule)
        # else:
        #     print("[Warning] No best checkpoint found. Skipping test evaluation.")

    def test_step(self, batch, batch_idx): 
        real_img, labels = batch 
        real_img = real_img.to(self.curr_device)
        labels = labels.to(self.curr_device)

        # Current epoch
        # current_epoch = self.trainer.max_epochs  
        
        # # Total number of epochs (from Trainer's max_epochs)
        # max_epoch = self.trainer.max_epochs

        results = self.forward(
            real_img,
            status = 'Valid',
            )   
        
        test_loss = self.model.loss_function(*results, M_N=1.0, batch_idx=batch_idx) 
 
        self.test_step_outputs.append(test_loss['loss']) 
        self.log_dict({f"test_{key}": test.item() for key, test in test_loss.items()}, sync_dist=True) 
 
        recon = results[0]
        real_img = real_img
        self.test_psnrs.append(self.psnr(recon, real_img)) 
        self.test_ssims.append(self.ssim(recon, real_img)) 
        self.test_mses.append(self.mse_loss_fn(recon, real_img))
        self.test_ms_ssims.append(msssim_fn(
            preds=recon, 
            target=real_img, 
            kernel_size=3, 
            betas=(0.4,0.4,0.2),
            data_range=1.0
        ))

        real_255 = (real_img * 255).clamp(0, 255).to(torch.uint8).to(self.curr_device)          # fid
        recon_255 = (recon * 255).clamp(0, 255).to(torch.uint8).to(self.curr_device)
        self.fid_metric.update(real_255, real=True)
        self.fid_metric.update(recon_255, real=False)

        sigma2_tensor = torch.tensor(1.0, device=self.curr_device)
        const_term = 0.5 * torch.log(2 * math.pi * sigma2_tensor)

        pixel_mse = (real_img-recon)**2                                                        # nlpd
        nlpd_map = const_term + pixel_mse/(2*sigma2_tensor)
        nlpd_sample = nlpd_map.mean()
        self.test_nlpds.append(nlpd_sample.cpu())

        with torch.no_grad():
            lpips_value = self.lpips_metrics(recon, real_img).mean()
        self.test_lpips.append(lpips_value)
        return test_loss['loss'] 

    # def on_test_epoch_end(self):
    #     # average
    #     avg_psnr = torch.stack(self.test_psnrs).mean() 
    #     avg_ssim = torch.stack(self.test_ssims).mean()
    #     avg_mse = torch.stack(self.test_mses).mean()
    #     fid_score = self.fid_metric.compute()
    #     avg_ms_ssim = torch.stack(self.test_ms_ssims).mean()
    #     avg_nlpd = torch.stack(self.test_nlpds).mean()
    #     avg_lpips = torch.stack(self.test_lpips).mean()

    #     self.test_psnrs.clear()
    #     self.test_ssims.clear()
    #     self.test_mses.clear()
    #     self.test_ms_ssims.clear()
    #     self.test_nlpds.clear()
    #     self.test_lpips.clear()
    #     self.fid_metric.reset()

    #     # log
    #     self.log("test_psnr", avg_psnr, prog_bar=True, sync_dist=True) 
    #     self.log("test_ssim", avg_ssim, prog_bar=True, sync_dist=True) 
    #     self.log("test_mse", avg_mse, prog_bar=True)
    #     self.log("test_fid", fid_score, prog_bar=True, sync_dist=True)
    #     self.log("test_ms_ssim", avg_ms_ssim, prog_bar=True, sync_dist=True)
    #     self.log("test_nlpd", avg_nlpd, prog_bar=True, sync_dist=True)
    #     self.log("test_lpips", avg_lpips, prog_bar=True, sync_dist=True)

    #     print(
    #         f"\n[Epoch {self.current_epoch}]"
    #         f"\ntest PSNR: {avg_psnr:.2f},"
    #         f"\ntest SSIM: {avg_ssim:.4f},"
    #         f"\ntest MSE: {avg_mse:.6f},"
    #         f"\ntest FID: {fid_score:.2f},"
    #         f"\ntest MS-SSIM: {avg_ms_ssim:.4f},"
    #         f"\ntest NLPD: {avg_nlpd:.4f},"
    #         f"\ntest LPIPS: {avg_lpips:.4f}") 
        
    #     metrics_path = os.path.join(self.logger.log_dir, "final_metrics_with_test.txt") 

    #     best_epoch_idx = int(np.argmax(self.val_ms_ssims_list))
    #     with open(metrics_path, "w") as f: 
    #         # Save alpha and beta if available
    #         try:
    #             alpha = self.model.alpha
    #             beta = self.model.beta
    #             soft_sample = self.model.vq_layer.soft_sample_flag
    #             q_flag = self.model.vq_layer.q_flag
    #             mem_flag = self.model.vq_layer.mem_flag
    #             random_flag = self.model.vq_layer.random_flag
    #             random_decay_method = self.model.vq_layer.decay_method          # New

    #             if not soft_sample and not q_flag and not mem_flag and not random_flag:
    #                 f.write("Traditional VQ-VAE method: \n")
    #             else:
    #                 f.write("our method: \n")
    #             f.write(f"{'Alpha':<15} {'-':>12} {alpha:>12.4f}\n")
    #             f.write(f"{'Beta':<15} {'-':>12} {beta:>12.4f}\n")
    #             f.write(f"{'q_flag':<15} {'-':>12} {str(q_flag):>12}\n")  # keep as float
    #             f.write(f"{'soft_sample':<15} {'-':>12} {str(soft_sample):>12}\n")
    #             f.write(f"{'mem_flag':<15} {'-':>12} {str(mem_flag):>12}\n")
    #             f.write(f"{'random_flag':<15} {'-':>12} {str(random_flag):>12}\n")
    #             f.write(f"{'random_decay_method':<15} {'-':>12} {str(random_decay_method):>12}\n")
    #         except AttributeError:
    #             print("[Warning] Alpha and Beta attributes not found in model.")
    #         f.write("[Final Epoch Metrics]\n") 
    #         f.write(f"{'Metric':<15} {'Train':>12} {'Validation':>12} {'Test':>12}\n") 
    #         f.write("-" * 55 + "\n") 
    #         f.write("{:<15}{:>12.6f}{:>12.6f}{:>12}\n".format(
    #                 "Loss",
    #                 self.train_losses[best_epoch_idx],
    #                 self.val_losses[best_epoch_idx],
    #                 '-')
    #         )
    #         f.write("{:<15}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(
    #                 'PSNR',
    #                 self.train_psnrs_cross_epochs[best_epoch_idx],
    #                 self.val_psnrs_list[best_epoch_idx],
    #                 avg_psnr.item())
    #         )
    #         f.write("{:<15}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(
    #                 'SSIM',
    #                 self.train_ssims_cross_epochs[best_epoch_idx],
    #                 self.val_ssims_list[best_epoch_idx],
    #                 avg_ssim.item())
    #         ) 
    #         f.write("{:<15}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(
    #                 'MSE',
    #                 self.train_mses_cross_epochs[best_epoch_idx],
    #                 self.val_mses_list[best_epoch_idx],
    #                 avg_mse.item())
    #         ) 
    #         f.write("{:<15}{:>12}{:>12.6f}{:>12.6f}\n".format(
    #                 'FID',
    #                 '-',
    #                 self.val_fids_list[best_epoch_idx],
    #                 fid_score.item())
    #         ) 
    #         f.write("{:<15}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(
    #                 'MS-SSIM',
    #                 self.train_ms_ssims_cross_epochs[best_epoch_idx],
    #                 self.val_ms_ssims_list[best_epoch_idx],
    #                 avg_ms_ssim.item())
    #         ) 
    #         f.write("{:<15}{:>12.6f}{:>12.6f}{:>12.6f}\n".format(
    #                 'NLPD',
    #                 self.train_nlpds_cross_epochs[best_epoch_idx],
    #                 self.val_nlpds_list[best_epoch_idx],
    #                 avg_nlpd.item()) 
    #         )
    #         f.write("{:<15}{:>12}{:>12.6f}{:>12.6f}\n".format(
    #                 'LPIPS',
    #                 '-',
    #                 self.val_lpips_list[best_epoch_idx],
    #                 avg_lpips.item())
    #         ) 

    def on_test_epoch_end(self):
        # --- compute test averages exactly as you already do ---
        avg_psnr = torch.stack(self.test_psnrs).mean()
        avg_ssim = torch.stack(self.test_ssims).mean()
        avg_mse = torch.stack(self.test_mses).mean()
        fid_score = self.fid_metric.compute()
        avg_ms_ssim = torch.stack(self.test_ms_ssims).mean()
        avg_nlpd = torch.stack(self.test_nlpds).mean()
        avg_lpips = torch.stack(self.test_lpips).mean()

        self.test_psnrs.clear()
        self.test_ssims.clear()
        self.test_mses.clear()
        self.test_ms_ssims.clear()
        self.test_nlpds.clear()
        self.test_lpips.clear()
        self.fid_metric.reset()

        self.log("test_psnr", avg_psnr, prog_bar=True, sync_dist=True)
        self.log("test_ssim", avg_ssim, prog_bar=True, sync_dist=True)
        self.log("test_mse", avg_mse, prog_bar=True)
        self.log("test_fid", fid_score, prog_bar=True, sync_dist=True)
        self.log("test_ms_ssim", avg_ms_ssim, prog_bar=True, sync_dist=True)
        self.log("test_nlpd", avg_nlpd, prog_bar=True, sync_dist=True)
        self.log("test_lpips", avg_lpips, prog_bar=True, sync_dist=True)

        print(
            f"\n[Epoch {self.current_epoch}]"
            f"\ntest PSNR: {avg_psnr:.2f},"
            f"\ntest SSIM: {avg_ssim:.4f},"
            f"\ntest MSE: {avg_mse:.6f},"
            f"\ntest FID: {fid_score:.2f},"
            f"\ntest MS-SSIM: {avg_ms_ssim:.4f},"
            f"\ntest NLPD: {avg_nlpd:.4f},"
            f"\ntest LPIPS: {avg_lpips:.4f}"
        )

        # --------- SAFE WRITE-OUT (works even when no train/val history is present) ----------
        metrics_path = os.path.join(self.logger.log_dir, "final_metrics_with_test.txt")

        # Helper: pick value from list if available, else None
        def pick(lst, idx):
            return lst[idx] if (idx is not None and 0 <= idx < len(lst)) else None

        # Pick best epoch by val MS-SSIM if we actually have validation history
        best_epoch_idx = int(np.argmax(self.val_ms_ssims_list)) if len(self.val_ms_ssims_list) > 0 else None

        # Helper: format either float tensor/number or '-'
        def fmt(x, nd=6):
            if x is None:
                return "-"
            if torch.is_tensor(x):
                x = x.item()
            return f"{x:.{nd}f}"

        with open(metrics_path, "w") as f:
            # (Optional block about alpha/beta/flags; keep your try/except as-is if you want)
            try:
                alpha = getattr(self.model, "alpha", None)
                beta = getattr(self.model, "beta", None)
                vq = getattr(self.model, "vq_layer", None)
                if vq is not None:
                    soft_sample = getattr(vq, "soft_sample_flag", False)
                    q_flag = getattr(vq, "q_flag", False)
                    mem_flag = getattr(vq, "mem_flag", False)
                    random_flag = getattr(vq, "random_flag", False)
                    random_decay_method = getattr(vq, "decay_method", None)
                else:
                    soft_sample = q_flag = mem_flag = random_flag = False
                    random_decay_method = None

                f.write("Traditional VQ-VAE method:\n" if not (soft_sample or q_flag or mem_flag or random_flag) else "our method:\n")
                if alpha is not None: f.write(f"{'Alpha':<15} {'-':>12} {fmt(alpha):>12}\n")
                if beta is not None:  f.write(f"{'Beta':<15} {'-':>12} {fmt(beta):>12}\n")
                f.write(f"{'q_flag':<15} {'-':>12} {str(q_flag):>12}\n")
                f.write(f"{'soft_sample':<15} {'-':>12} {str(soft_sample):>12}\n")
                f.write(f"{'mem_flag':<15} {'-':>12} {str(mem_flag):>12}\n")
                f.write(f"{'random_flag':<15} {'-':>12} {str(random_flag):>12}\n")
                f.write(f"{'random_decay_method':<15} {'-':>12} {str(random_decay_method):>12}\n")
            except Exception:
                pass

            f.write("[Final Epoch Metrics]\n")
            f.write(f"{'Metric':<15} {'Train':>12} {'Validation':>12} {'Test':>12}\n")
            f.write("-" * 55 + "\n")

            # Loss
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "Loss",
                fmt(pick(self.train_losses, best_epoch_idx)),
                fmt(pick(self.val_losses, best_epoch_idx)),
                "-"  # no single test loss aggregate here
            ))
            # PSNR
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "PSNR",
                fmt(pick(self.train_psnrs_cross_epochs, best_epoch_idx), nd=2),
                fmt(pick(self.val_psnrs_list, best_epoch_idx), nd=2),
                fmt(avg_psnr, nd=2)
            ))
            # SSIM
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "SSIM",
                fmt(pick(self.train_ssims_cross_epochs, best_epoch_idx)),
                fmt(pick(self.val_ssims_list, best_epoch_idx)),
                fmt(avg_ssim)
            ))
            # MSE
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "MSE",
                fmt(pick(self.train_mses_cross_epochs, best_epoch_idx)),
                fmt(pick(self.val_mses_list, best_epoch_idx)),
                fmt(avg_mse)
            ))
            # FID
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "FID",
                "-",
                fmt(pick(self.val_fids_list, best_epoch_idx), nd=2),
                fmt(fid_score, nd=2)
            ))
            # MS-SSIM
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "MS-SSIM",
                fmt(pick(self.train_ms_ssims_cross_epochs, best_epoch_idx)),
                fmt(pick(self.val_ms_ssims_list, best_epoch_idx)),
                fmt(avg_ms_ssim)
            ))
            # NLPD
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "NLPD",
                fmt(pick(self.train_nlpds_cross_epochs, best_epoch_idx)),
                fmt(pick(self.val_nlpds_list, best_epoch_idx)),
                fmt(avg_nlpd)
            ))
            # LPIPS
            f.write("{:<15}{:>12}{:>12}{:>12}\n".format(
                "LPIPS",
                "-",
                fmt(pick(self.val_lpips_list, best_epoch_idx)),
                fmt(avg_lpips)
            ))
