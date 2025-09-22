#%%
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *                  # import from model/types_.py
import math
#%%
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int,
                beta: float = 0.25,
                alpha_q: float = 1,

                q_flag: bool = False,

                random_flag: bool = False,
                random_prob: float = 0.0,
                random_decay_method: str = "constant",
                decay_kwargs: dict = None,
                
                soft_sample_flag: bool = False,
                mem_flag: bool = False): # default p = 0
        
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.q_flag = q_flag
        self.random_flag = random_flag
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        self.alpha = alpha_q
        self.random_prob = random_prob  # initial probability
        self.decay_method = random_decay_method
        self.decay_kwargs = decay_kwargs or {}  # hyperparameters for decay

        self.soft_sample_flag = soft_sample_flag
        self.mem_flag = mem_flag and (not soft_sample_flag)

    def forward(self, latents: Tensor, epoch: int = None, max_epoch: int = None, status: str = 'Train') -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        if self.soft_sample_flag and status =='Train' and epoch is not None and max_epoch is not None:
            tau_init = 1.0
            tau = self.get_decay_p(
                    method='exp', 
                    epoch=epoch, 
                    max_epoch=max_epoch, 
                    initial_p=tau_init,
                    **{'decay_rate': 0.80} )
            encoding_inds = self.soft_sample(dist, tau=tau) #[BWH,1]    
            encoding_inds = encoding_inds.unsqueeze(1)  # [BHW]
            # Convert to one-hot encodings
            device = latents.device
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
            encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # elif self.mem_flag and status == 'Train' and epoch is not None and max_epoch is not None:
        #     tau_init = 1
        #     tau = self.get_decay_p(
        #             method='exp', 
        #             epoch=epoch, 
        #             max_epoch=max_epoch, 
        #             initial_p=tau_init,
        #             warmup_flag=True,
        #             warmup_epochs=2,
        #             warmup_type='cosine',
        #             peak_p=1,
        #             **{'decay_rate': 0.80} )
        #     encoding_one_hot = self.mem_sample(dist, tau=tau).detach() #[BWH,k]
        #     encoding_inds = torch.argmax(encoding_one_hot, dim=1, keepdim=True).unsqueeze(1)  # [BHW]

        elif self.random_flag and epoch is not None and max_epoch is not None:
            # Compute probability p with decay schedule
            p = self.random_prob
            p = self.get_decay_p(
                method=self.decay_method,
                epoch=epoch,
                max_epoch=max_epoch,
                initial_p=self.random_prob,
                **self.decay_kwargs
            )
            # Random sampling mask
            encoding_inds = torch.argmin(dist, dim=1)  # [BHW, 1]  
            if p > 0 and status == 'Train':
                mask = (torch.rand_like(encoding_inds.float()) < p)
                random_inds = torch.randint(0, self.K, encoding_inds.shape, device=latents.device)
                encoding_inds = torch.where(mask, random_inds, encoding_inds)    
            encoding_inds = encoding_inds.unsqueeze(1)  # [BHW]
            # Convert to one-hot encodings
            device = latents.device
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
            encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        else:
            encoding_inds = torch.argmin(dist, dim=1)  # [BHW, 1]      
            encoding_inds = encoding_inds.unsqueeze(1)  # [BHW]

            # Convert to one-hot encodings
            device = latents.device
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
            encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        if self.mem_flag and status == 'Train' and epoch is not None and max_epoch is not None:
            tau_init = 0.05
            tau = self.get_decay_p(
                    method='exp', 
                    epoch=epoch, 
                    max_epoch=max_epoch, 
                    initial_p=tau_init,
                    warmup_flag=False,
                    warmup_epochs=2,
                    warmup_type='cosine',
                    peak_p=1,
                    **{'decay_rate': 0.90} )
            embedding_loss_dist_weight = self.mem_sample(dist, tau=tau).detach() #[BWH,k]

            # Compute squared distance to all embeddings
            diff = flat_latents.unsqueeze(1) - self.embedding.weight.unsqueeze(0)  # [BHW, K, D]
            squared_error = diff.pow(2)  # [BHW, K, D]
            mse_per_embedding = squared_error.mean(dim=-1)  # [BHW, K]

            # Weighted embedding loss
            embedding_loss = (embedding_loss_dist_weight * mse_per_embedding).sum() / flat_latents.shape[0]
        else:
            embedding_loss = F.mse_loss(quantized_latents, latents.detach()) # [B x H x W x D] vs [B x H x W x D]

        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        vq_loss = commitment_loss * self.beta + embedding_loss * self.alpha
        if self.q_flag: 
            alpha_q = 2 * self.alpha / (latents_shape[0] * latents_shape[1] * latents_shape[2] * latents_shape[3])
            quantized_latents = latents + alpha_q * quantized_latents + ((1-alpha_q)*quantized_latents - latents).detach() # method 3, corrected formula
        # elif self.mem_flag and status =='Train' and epoch is not None and max_epoch is not None:
        #     tau_init = 0.05
        #     tau = self.get_decay_p(
        #             method='exp', 
        #             epoch=epoch, 
        #             max_epoch=max_epoch, 
        #             initial_p=tau_init,
        #             warmup_flag=False,
        #             warmup_epochs=2,
        #             warmup_type='cosine',
        #             peak_p=1,
        #             **{'decay_rate': 0.90} )
        #     sum_vec = self.embedding.weight.sum(dim=0)
        #     out = sum_vec.view(1, 1, 1, -1).expand(latents_shape[0], latents_shape[1], latents_shape[2], -1)  # [B,H,W,D]
        #     quantized_latents = latents + tau * out + (quantized_latents - latents - tau * out).detach() # method 3, corrected formula
        else:

             # Add the residue back to the latents
            quantized_latents = latents + (quantized_latents - latents).detach()         # standard method 0
            
        return (  
            quantized_latents.permute(0, 3, 1, 2).contiguous(), # [B x D x H x W]
            vq_loss, 
            encoding_inds.view(latents_shape[0], latents_shape[1], latents_shape[2])
          )
    
    def soft_sample(self, dist, tau=1.0):
        """
        给定距离矩阵 dist, 按 softmax(-dist / tau) 采样每个样本的码字索引。

        参数：
        - dist: [B, K], B 个样本与 K 个码字的距离（越小越近）
        - tau: 温度系数，控制 softmax 分布平滑程度

        返回：
        - indices: [B]，采样得到的码字索引
        """
        # 计算概率门控
        logits = -dist / tau                     # 越近的距离 → logit 越大
        probs = F.softmax(logits, dim=1)         # [BHW, K]

        # 按概率分布采样一个索引
        indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [BHW, 1]

        return indices

    def mem_sample(self, dist, tau=1.0):
        logits = F.softmax(-dist/tau, dim = 1) # [BHW, K]
        return logits

    def get_decay_p(
            self, 
            method: str, 
            epoch: int, 
            max_epoch: int, 
            initial_p: float = 0.1, 
            
            warmup_flag: bool = False,
            warmup_epochs: int = 0,
            warmup_type: str = 'linear',
            peak_p: float = 0.5,

            **kwargs) -> float:
        """
        Compute probability p with different decay strategies.
        
        Parameters
        ----------
        method : str
            Name of the decay method: 
            ["None", "linear", "exp", "cosine", "step", "inverse", "poly", "sigmoid", "cyclic"]
        epoch : int
            Current epoch (>=0)
        max_epoch : int
            Maximum epoch
        initial_p : float
            Starting probability
        **kwargs : dict
            Extra hyperparameters for each method
        
        If warmup_flag is True:
        - Warmup phase (0 .. warmup_epochs-1): increase from ~0 to peak_p
        - Decay phase (warmup_epochs .. max_epoch): apply `method` over the
            remaining epochs with initial_p=peak_p so it's continuous
        Otherwise:
        - Behaves like the original function.

        Returns
        -------
        float
            Probability p (>=0)
        """
        def _base_schedule(method: str, epoch: int, max_epoch: int, initial_p: float) -> float:
            method = method.lower()
            p = initial_p

            if method == "constant":
                p = p

            elif method == "linear":
                p = initial_p * max(0.0, 1 - epoch / max_epoch)
                
            elif method == "exp":
                decay_rate = kwargs.get("decay_rate", 0.99)
                print(f"decay_rate:{decay_rate}, epoch:{epoch}")
                p = initial_p * (decay_rate ** epoch)
                
            elif method == "cosine":
                p = initial_p * 0.5 * (1 + math.cos(math.pi * epoch / max_epoch))
                
            elif method == "step":
                decay_factor = kwargs.get("decay_factor", 0.5)
                step_size = kwargs.get("step_size", max(1, max_epoch // 3))
                p = initial_p * (decay_factor ** (epoch // step_size))
                
            elif method == "inverse":
                k = kwargs.get("k", 5.0)
                p = initial_p / (1 + k * epoch / max_epoch)
                
            elif method == "poly":
                k = kwargs.get("k", 2.0)  # 2.0 = quadratic
                p = initial_p * (1 - epoch / max_epoch) ** k
                
            elif method == "sigmoid":
                center = kwargs.get("center", max_epoch * 0.5)
                sharpness = kwargs.get("sharpness", 0.1)
                p = initial_p / (1 + math.exp((epoch - center) * sharpness))
                
            elif method == "cyclic":
                num_cycles = kwargs.get("num_cycles", 3)
                cycle_length = max_epoch // num_cycles
                phase = epoch % cycle_length
                p = initial_p * 0.5 * (1 + math.cos(math.pi * phase / cycle_length))
                
            else:
                raise ValueError(f"Unknown decay method: {method}")
            
            return p

        # ------ Warmup wrapper ----------------
        if warmup_flag and warmup_epochs >0:
            if epoch < warmup_epochs:
                # Warmup from near 0 to peak_p
                if warmup_type.lower() == "cosine":
                    # smooth start (S-shaped): 0 -> peak_p
                    progress = epoch / warmup_epochs
                    p = peak_p * 0.5 * (1 - math.cos(math.pi * progress))
                else:
                    # linear warmup: 0 -> peak_p
                    p = peak_p * (epoch / warmup_epochs)
                return max(p, 0.00000001)
            else:
                # Decay phase with re-based timeline
                decay_epochs_total = max(1, max_epoch - warmup_epochs)
                decay_epoch = min(epoch - warmup_epochs, decay_epochs_total)  # clamp
                # Start the chosen method at peak_p and run for the remaining epochs
                p = _base_schedule(method, decay_epoch, decay_epochs_total, peak_p)
                return max(p, 0.000000001)
        else:
            # no warmup at all
            p = _base_schedule(method, epoch, max_epoch, initial_p)
            return max(p, 1e-8)
        
class PCA(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int = 1024,
                method: str = "qr",
                lr: float = 0.001
                ): # default p = 0
        
        super().__init__()
        self.Q = num_embeddings
        self.N = embedding_dim
        assert self.Q <= self.N, "Need Q ≤ N for orthonormal columns in R^D."

        self.base_lr = lr

        if method == "eye":
            # First Q columns of I_D → [N, Q]
            codes = torch.eye(self.N, self.Q)
        else:
            # Random N x Q, thin-QR → Qmat has orthonormal columns
            M = torch.randn(self.N, self.Q)
            Qmat, _ = torch.linalg.qr(M, mode="reduced")  # [N, Q], columns orthonormal
            codes = Qmat

        # Non-learnable, saved in state_dict, moves with .to(device)
        self.register_buffer("codes", codes)  # [N, Q]
        self.register_buffer("xN_mean", torch.zeros(self.N))
        # MLP will be built lazily after we know D*H*W
        self.mapper = None
        self.decoder = None       # [B, N]   -> [B, DHW]
        self._in_features = None
        
        # update counter for periodic orthonormalization
        self.register_buffer("_update_count", torch.zeros((), dtype=torch.long))
        # use float64 accumulator for numeric stability on counts
        self.register_buffer("_mean_count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def _update_mean(self, x_N: torch.Tensor, ema_alpha: float = 0.9):
        """
        Two-step mean update:
        1) exact online mean with batch size B
        2) EMA blend toward current batch mean
        """
        B = x_N.shape[0]
        if B == 0:
            return

        batch_mean = x_N.mean(dim=0)  # [N]
        # Step 1: exact online mean using counts
        total = self._mean_count + float(B)                  # scalar
        w = float(B) / float(total)                          # in (0,1]
        mu_exact = self.xN_mean + w * (batch_mean - self.xN_mean)

        # Step 2: EMA blend toward the batch mean
        mu_blend = ema_alpha * mu_exact + (1.0 - ema_alpha) * batch_mean

        # Commit
        self.xN_mean.copy_(mu_blend)
        self._mean_count.copy_(total)

    def _build_heads(self, in_features: int, device: torch.device):
        # Build both mapper and decoder once DHW is known
        hidden_enc = max(64, min(4 * self.N, 1024))
        hidden_dec = max(64, min(4 * in_features, 1024))

        self.mapper = nn.Sequential(
            nn.Linear(in_features, hidden_enc),
            nn.GELU(),
            nn.Linear(hidden_enc, self.N),  # -> [B, N]
        ).to(device)   # <- ensure on the right device

        self.decoder = nn.Sequential(
            nn.Linear(self.N, hidden_dec),
            nn.GELU(),
            nn.Linear(hidden_dec, in_features),  # -> [B, DHW]
        ).to(device)   # <- ensure on the right device

        self._in_features = in_features

    @torch.no_grad()
    def _orthonormalize_sym(self, eps: float = 1e-8):
        """
        Symmetric normalization: C <- C (C^T C)^{-1/2}
        Makes columns ~orthonormal even if C drifted.
        """
        # G = C^T C  (Q x Q, SPD)
        G = self.codes.T @ self.codes                      # [Q, Q]
        # eigendecomp (symmetric): G = U diag(lam) U^T
        lam, U = torch.linalg.eigh(G)                      # lam: [Q], U: [Q, Q]
        # inverse sqrt of diag(lam) with clamp for stability
        inv_sqrt_lam = torch.clamp(lam, min=eps).rsqrt()   # 1/sqrt(lam)
        S_inv = (U * inv_sqrt_lam) @ U.T                   # U diag(inv_sqrt_lam) U^T, [Q, Q]
        # C <- C S^{-1}
        self.codes.copy_(self.codes @ S_inv)

    @torch.no_grad()
    def _codes_update(self, x_N: Tensor, lr: float):
        """
        Oja-like update on orthonormal codes with re-orthonormalization.
        x_N: [B, N]
        codes: [N, Q]
        """
        B = x_N.shape[0]
        # coefficients Y = X C, projection X_hat = Y C^T
        Y = x_N @ self.codes                       # [B, Q]
        # gradient-like term: X^T Y - C (Y^T Y)   (matrix sizes match [N,Q])
        # scale by 1/B for batch-size invariance
        dC = (x_N.T @ Y - self.codes @ (Y.T @ Y)) / max(1, B)
        # in-place update on the buffer
        self.codes.add_(lr * dC)

        # periodic maintenance: every Q updates
        self._update_count.add_(1)
        if int(self._update_count.item()) % self.Q == 0:
            self._orthonormalize_sym()


    def forward(self, latents: Tensor, epoch: int = None, max_epoch: int = None, status: str = 'Train'):
        """
        latents: [B, D, H, W]
        Returns:
          x_hat:      [B, D, H, W]     (reconstruction in input shape)
          coeffs_Y:   [B, Q]           (low-dim coefficients in code space)
          proj_in_N:  [B, N]           (projection/reconstruction in N-dim space)
        """
        assert latents.dim() == 4, "Expect latents as [B, D, H, W]"
        B, D, H, W = latents.shape
        DHW = D * H * W
        device = latents.device
        # [B, DHW]
        x = latents.view(B, DHW)

        # Build MLPs lazily for this DHW
        if (self.mapper is None) or (self._in_features != DHW):
            self._build_heads(DHW, device)

        # Map to N
        x_N = self.mapper(x)                      # [B, N]

        # --- NEW: update running mean during training, then center ---
        if status == 'Train':
            # Update mean with current batch BEFORE centering
            self._update_mean(x_N.detach())
        
        x_N_centered = x_N - self.xN_mean  # [B, N], gradients flow through x_N

        # update codes only in training and when scheduler params are provided
        if status == 'Train' and (epoch is not None) and (max_epoch is not None):
            lr_t = self.get_decay(method="cosine", epoch=epoch, max_epoch=max_epoch, initial_p=self.base_lr)
            self._codes_update(x_N_centered.detach(), lr_t)

        # Project to Q (coefficients) and back to N using codes (orthonormal)
        # codes: [N, Q]
        Y = x_N_centered @ self.codes                      # [B, Q]
        x_rec_centered = Y @ self.codes.T                # [B, N] (projection in N-dim)

        # Add mean back to return to original N-space
        x_rec_N = x_rec_centered + self.xN_mean       # [B, N]

        # Decode back to DHW and reshape
        x_hat = self.decoder(x_rec_N)             # [B, DHW]
        x_hat = x_hat.view(B, D, H, W)            # [B, D, H, W]

        return x_hat, torch.tensor(0., device=latents.device), torch.zeros((B, H, W), dtype=torch.long, device=device)
    
    def get_decay(
            self, 
            method: str, 
            epoch: int, 
            max_epoch: int, 
            initial_p: float = 0.1, 
            
            warmup_flag: bool = False,
            warmup_epochs: int = 0,
            warmup_type: str = 'linear',
            peak_p: float = 0.5,

            **kwargs) -> float:
        """
        Compute probability p with different decay strategies.
        
        Parameters
        ----------
        method : str
            Name of the decay method: 
            ["None", "linear", "exp", "cosine", "step", "inverse", "poly", "sigmoid", "cyclic"]
        epoch : int
            Current epoch (>=0)
        max_epoch : int
            Maximum epoch
        initial_p : float
            Starting probability
        **kwargs : dict
            Extra hyperparameters for each method
        
        If warmup_flag is True:
        - Warmup phase (0 .. warmup_epochs-1): increase from ~0 to peak_p
        - Decay phase (warmup_epochs .. max_epoch): apply `method` over the
            remaining epochs with initial_p=peak_p so it's continuous
        Otherwise:
        - Behaves like the original function.

        Returns
        -------
        float
            Probability p (>=0)
        """
        def _base_schedule(method: str, epoch: int, max_epoch: int, initial_p: float) -> float:
            method = method.lower()
            p = initial_p

            if method == "constant":
                p = p

            elif method == "linear":
                p = initial_p * max(0.0, 1 - epoch / max_epoch)
                
            elif method == "exp":
                decay_rate = kwargs.get("decay_rate", 0.99)
                print(f"decay_rate:{decay_rate}, epoch:{epoch}")
                p = initial_p * (decay_rate ** epoch)
                
            elif method == "cosine":
                p = initial_p * 0.5 * (1 + math.cos(math.pi * epoch / max_epoch))
                
            elif method == "step":
                decay_factor = kwargs.get("decay_factor", 0.5)
                step_size = kwargs.get("step_size", max(1, max_epoch // 3))
                p = initial_p * (decay_factor ** (epoch // step_size))
                
            elif method == "inverse":
                k = kwargs.get("k", 5.0)
                p = initial_p / (1 + k * epoch / max_epoch)
                
            elif method == "poly":
                k = kwargs.get("k", 2.0)  # 2.0 = quadratic
                p = initial_p * (1 - epoch / max_epoch) ** k
                
            elif method == "sigmoid":
                center = kwargs.get("center", max_epoch * 0.5)
                sharpness = kwargs.get("sharpness", 0.1)
                p = initial_p / (1 + math.exp((epoch - center) * sharpness))
                
            elif method == "cyclic":
                num_cycles = kwargs.get("num_cycles", 3)
                cycle_length = max_epoch // num_cycles
                phase = epoch % cycle_length
                p = initial_p * 0.5 * (1 + math.cos(math.pi * phase / cycle_length))
                
            else:
                raise ValueError(f"Unknown decay method: {method}")
            
            return p

        if warmup_flag and warmup_epochs > 0:
            if epoch < warmup_epochs:
                progress = epoch / warmup_epochs
                if warmup_type.lower() == "cosine":
                    # cosine ramp-up
                    p = peak_p * 0.5 * (1 - math.cos(math.pi * progress))
                else:
                    # linear ramp-up
                    p = peak_p * progress
                return max(p, 1e-8)
            else:
                # After warmup → use base schedule starting from peak_p
                decay_epochs_total = max(1, max_epoch - warmup_epochs)
                decay_epoch = min(epoch - warmup_epochs, decay_epochs_total)
                p = _base_schedule(method, decay_epoch, decay_epochs_total, peak_p)
                return max(p, 1e-8)
        else:
            # no warmup at all
            p = _base_schedule(method, epoch, max_epoch, initial_p)
            return max(p, 1e-8)
    
class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                in_channels: int,
                hidden_dims: List = None,
                img_size: int = 64,

                embedding_dim: int = 64,
                num_embeddings: int = 64,
                vq_flag: bool = False,
                vq_parameters: dict = None,
                **kwargs) -> None:
        
        super(VQVAE, self).__init__()

        beta = vq_parameters.get("beta", 0.25)
        alpha = vq_parameters.get("alpha", 1)
        q_flag = vq_parameters.get("q_flag", False)
        random_flag = vq_parameters.get("random_flag", False)
        random_prob = vq_parameters.get("random_prob", 0.0)
        random_decay_method = vq_parameters.get("random_decay_method", "constant")
        decay_kwargs = vq_parameters.get("decay_kwargs", None)
        soft_sample_flag = vq_parameters.get("soft_sample_flag", False)
        mem_flag = vq_parameters.get("mem_flag", False)
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.alpha = alpha

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        # ---------------------------
        # Vector Quantization Layer
        # ---------------------------
        self.vq_flag = vq_flag
        if self.vq_flag:
            self.vq_layer = VectorQuantizer(
                num_embeddings = num_embeddings,
                embedding_dim = embedding_dim,
                beta = beta,
                alpha_q = alpha,

                q_flag = q_flag,

                random_flag = random_flag,
                random_prob = random_prob,
                random_decay_method = random_decay_method,
                decay_kwargs = decay_kwargs,       # pass decay hyperparameters

                soft_sample_flag = soft_sample_flag,
                mem_flag = mem_flag
            )
        else:
            self.vq_layer = PCA(
                num_embeddings= num_embeddings,
                )
        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, epoch: int = None, max_epoch: int = None, status: str = 'Train', **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]

        quantized_inputs, vq_loss, encoding_inds = self.vq_layer(encoding, epoch, max_epoch, status)

        return [self.decode(quantized_inputs), input, vq_loss, encoding_inds]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    # def sample(self,
    #            num_samples: int,
    #            current_device: Union[int, str], **kwargs) -> Tensor:
    #     # raise Warning('VQVAE sampler is not implemented.')                      # don't need
    #     pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, status='Valid')[0]             # take reconstructed img

#