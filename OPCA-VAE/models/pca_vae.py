#%%
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *                  # import from model/types_.py
import math

        
class PCA(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int | None = 1024,
                method: str = "qr",
                # lr: float = 0.001
                ): # default p = 0
        
        super().__init__()
        self.Q = num_embeddings
        self.N = embedding_dim
        # assert self.Q <= self.N, "Need Q ≤ N for orthonormal columns in R^D."
        self._use_mapper = embedding_dim is not None
        self._init_method = method
        # self.base_lr = lr

        # if method == "eye":
        #     # First Q columns of I_D → [N, Q]
        #     codes = torch.eye(self.N, self.Q)
        # else:
        #     # Random N x Q, thin-QR → Qmat has orthonormal columns
        #     M = torch.randn(self.N, self.Q)
        #     Qmat, _ = torch.linalg.qr(M, mode="reduced")  # [N, Q], columns orthonormal
        #     codes = Qmat

        # Non-learnable, saved in state_dict, moves with .to(device)
        # self.register_buffer("codes", codes)  # [N, Q]
        # self.register_buffer("xN_mean", torch.zeros(self.N))
        # MLP will be built lazily after we know D*H*W
        self.mapper = None
        self.decoder = None       # [B, N]   -> [B, DHW]
        self._in_features = None
        
        # update counter for periodic orthonormalization
        self.register_buffer("_update_count", torch.zeros((), dtype=torch.long))
        # use float64 accumulator for numeric stability on counts
        self.register_buffer("_mean_count", torch.zeros((), dtype=torch.float64))

    def _init_codes_if_needed(self, N: int, device: torch.device):
        if hasattr(self, "codes"):
            return  # already built
        assert self.Q <= N, f"Need Q ≤ N for orthonormal columns in R^D (Q={self.Q}, N={N})."
        if self._init_method == "eye":
            codes = torch.eye(N, self.Q, device=device)
        else:
            M = torch.randn(N, self.Q, device=device)
            Qmat, _ = torch.linalg.qr(M, mode="reduced")  # [N,Q]
            codes = Qmat
        self.register_buffer("codes", codes)   # [N,Q]
        self.register_buffer("xN_mean", torch.zeros(N, device=device))
        # self.register_buffer("_update_count", torch.zeros((), dtype=torch.long))
        # self.register_buffer("_mean_count", torch.zeros((), dtype=torch.float64))
        self.N = N

    def _ensure_codes(self, N_target: int, device: torch.device):
        if not hasattr(self, "codes"):
            self._init_codes_if_needed(N_target, device)
        else:
            if self.codes.shape[0] != N_target:
                raise RuntimeError(
                    f"N mismatch: codes built for N={self.codes.shape[0]} but current N={N_target}. "
                    "If you use direct mode (embedding_dim=None), DHW must stay constant."
                )
            
    @torch.no_grad()
    def _update_mean(self, x_N: torch.Tensor, ema_alpha: float = 0.9, ):
        """
        
        """
        B = x_N.shape[0]
        if B == 0:
            return
        # Align device/dtype with the stored mean buffer to avoid device mismatches.

        batch_mean = x_N.mean(dim=0)  # [N]
        # Step 1: exact online mean using counts
        n = self._mean_count                  # scalar
        S_n = (1 - ema_alpha ** n)/(1 - ema_alpha)
        S_n1 = (1 - ema_alpha ** (n+1))/(1 - ema_alpha)
        mu_exact = 1/S_n1 * batch_mean + ema_alpha*S_n/S_n1 * self.xN_mean

        # Commit
        self.xN_mean.copy_(mu_exact)
        self._mean_count.copy_(n+1)

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
        gram = Y.T @ Y                                      # [Q, Q]
        gram_upper = torch.triu(gram)                       # zero out subdiagonal
        # gradient-like term: X^T Y - C upper(Y^T Y)   (matrix sizes match [N,Q])
        # scale by 1/B for batch-size invariance
        
        dC = (x_N.T @ Y - self.codes @ gram_upper ) / max(1, B)
        # in-place update on the buffer
        self.codes.add_(lr * dC)

        # periodic maintenance: every Q updates
        self._update_count.add_(1)
        if int(self._update_count.item()) % self.Q == 0:
            self._orthonormalize_sym()


    def forward(self, 
                latents: Tensor, 
                # epoch: int = None, 
                # max_epoch: int = None,
                status: str = 'Train', 
                current_lr: float|None = None
                ):
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

        # Decide the target N and ensure codes/xN_mean exist for BOTH modes
        N_target = (self.N if (getattr(self, "_use_mapper", False) and self.N is not None) else DHW)
        self._ensure_codes(N_target, device)

        if self._use_mapper:  
            # --- Mapper mode ---
            if (self.mapper is None) or (self._in_features != DHW):
                self._build_heads(DHW, device)
            x_N = self.mapper(x)
        else:
            # --- Direct mode ---
            x_N = x

        # # Build MLPs lazily for this DHW
        # if (self.mapper is None) or (self._in_features != DHW):
        #     self._build_heads(DHW, device)

        # Map to N
        # x_N = self.mapper(x)                      # [B, N]

        # --- NEW: update running mean during training, then center ---
        if status == 'Train':
            # Update mean with current batch BEFORE centering
            self._update_mean(x_N.detach())
        
        x_N_centered = x_N - self.xN_mean  # [B, N], gradients flow through x_N

        # update codes only in training and when scheduler params are provided
        if status == 'Train' and current_lr is not None:
            # lr_t = self.get_decay(method="cosine", epoch=epoch, max_epoch=max_epoch, initial_p=self.base_lr)
            self._codes_update(x_N_centered.detach(), current_lr)

        # Project to Q (coefficients) and back to N using codes (orthonormal)
        # codes: [N, Q]
        Y = x_N_centered @ self.codes                      # [B, Q]
        x_rec_centered = Y @ self.codes.T                # [B, N] (projection in N-dim)

        # Add mean back to return to original N-space
        x_rec_N = x_rec_centered + self.xN_mean       # [B, N]

        # Decode back to DHW and reshape
        if self._use_mapper:
            x_hat = self.decoder(x_rec_N)             # [B, DHW]
        else:
            x_hat = x_rec_N
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


class PCAVAE(BaseVAE):

    def __init__(self,
                in_channels: int,
                hidden_dims: List = None,
                img_size: int = 64,

                embedding_dim: int = 64,
                num_embeddings: int = 64,
                method: str = 'qr', # it can be 'eye' or 'qr'
                **kwargs) -> None:
        
        super(PCAVAE, self).__init__()

        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size


        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 64, 64]

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
        # PCA Layer
        # ---------------------------
      
        self.vq_layer = PCA(
            num_embeddings= num_embeddings,
            embedding_dim= None,
            method = "qr",
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

    def forward(self, input: Tensor, status: str = 'Train', current_lr: float = None, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]

        quantized_inputs, vq_loss, encoding_inds = self.vq_layer(
            latents = encoding, 
            status = status,
            current_lr = current_lr)

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