#%%
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *                  # import from model/types_.py
import math

from typing import Optional
import torch
from torch import nn, Tensor
from utils import instantiate_from_config

class PCA(nn.Module):
    """
    PCA quantizer with two modes:
      - keep_shape=False (default): flatten [B,D,H,W] -> [B,DHW], (optional) map to N,
        learn a single [N,Q] codebook.
      - keep_shape=True: run H*W parallel PCA quantizers over channel vectors [B,D] at
        each spatial location; per-location codebooks [L,D,Q] with L=H*W.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: Optional[int] = 1024,
                 method: str = "qr",
                 keep_shape: bool = False):
        super().__init__()
        self.Q = num_embeddings
        self.N = embedding_dim
        self.keep_shape = keep_shape
        print(self.Q, self.N, self.keep_shape)
        # In classic mode, allow N != None to enable mapper/decoder
        self._use_mapper = (embedding_dim is not None) and (keep_shape is False)
        self._init_method = method

        self.mapper = None
        self.decoder = None
        self._in_features = None

        # global (classic) counters
        self.register_buffer("_update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_mean_count", torch.zeros((), dtype=torch.float64))

        # spatial mode buffers are built lazily once H,W are known
        #   codes_spatial: [L, D, Q]
        #   x_mean_spatial: [L, D]
        #   mean_count_spatial: [L]
        self.register_buffer("_spatial_built", torch.tensor(False))
        # place-holders for static checks
        self._D_built_spatial = None
        self._L_built_spatial = None

    # ---------- Classic single-codebook mode (original behavior) ----------

    def _init_codes_if_needed(self, N: int, device: torch.device):
        if hasattr(self, "codes"):
            return
        assert self.Q <= N, f"Need Q ≤ N for orthonormal columns in R^N (Q={self.Q}, N={N})."
        if self._init_method == "eye":
            codes = torch.eye(N, self.Q, device=device)
        else:
            M = torch.randn(N, self.Q, device=device)
            Qmat, _ = torch.linalg.qr(M, mode="reduced")
            codes = Qmat
        self.register_buffer("codes", codes)                 # [N, Q]
        self.register_buffer("xN_mean", torch.zeros(N, device=device))
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
    def _update_mean(self, x_N: torch.Tensor, ema_alpha: float = 0.9):
        B = x_N.shape[0]
        if B == 0:
            return
        batch_mean = x_N.mean(dim=0)  # [N]
        n = self._mean_count
        S_n  = (1 - ema_alpha**n) / (1 - ema_alpha)
        S_n1 = (1 - ema_alpha**(n+1)) / (1 - ema_alpha)
        mu_exact = (batch_mean / S_n1) + (ema_alpha * S_n / S_n1) * self.xN_mean
        self.xN_mean.copy_(mu_exact)
        self._mean_count.copy_(n + 1)

    def _build_heads(self, in_features: int, device: torch.device):
        hidden_enc = max(64, min(4 * self.N, 1024))
        hidden_dec = max(64, min(4 * in_features, 1024))
        self.mapper = nn.Sequential(
            nn.Linear(in_features, hidden_enc),
            nn.GELU(),
            nn.Linear(hidden_enc, self.N),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(self.N, hidden_dec),
            nn.GELU(),
            nn.Linear(hidden_dec, in_features),
        ).to(device)
        self._in_features = in_features

    @torch.no_grad()
    def _orthonormalize_sym(self, eps: float = 1e-8):
        # C <- C (C^T C)^(-1/2)
        G = self.codes.T @ self.codes            # [Q,Q]
        lam, U = torch.linalg.eigh(G)            # SPD
        inv_sqrt_lam = torch.clamp(lam, min=eps).rsqrt()
        S_inv = (U * inv_sqrt_lam) @ U.T
        self.codes.copy_(self.codes @ S_inv)

    @torch.no_grad()
    def _codes_update(self, x_N: Tensor, lr: float):
        B = x_N.shape[0]
        Y = x_N @ self.codes                       # [B,Q]
        gram = Y.T @ Y                              # [Q,Q]
        gram_upper = torch.triu(gram)
        dC = (x_N.T @ Y - self.codes @ gram_upper) / max(1, B)
        self.codes.add_(lr * dC)
        self._update_count.add_(1)
        if int(self._update_count.item()) % self.Q == 0:
            self._orthonormalize_sym()

    # ---------- Spatial parallel PCA mode (new) ----------

    def _build_spatial_if_needed(self, D: int, H: int, W: int, device: torch.device):
        if bool(self._spatial_built):
            # static shape checks
            if (self._D_built_spatial != D) or (self._L_built_spatial != H * W):
                raise RuntimeError(
                    f"keep_shape mode was built for D={self._D_built_spatial}, L={self._L_built_spatial} "
                    f"but current D={D}, L={H*W}. Spatial size must be consistent."
                )
            return

        L = H * W
        assert self.Q <= D, f"In keep_shape=True, need Q ≤ D (Q={self.Q}, D={D})."

        # Initialize one orthonormal basis [D,Q], then replicate to [L,D,Q]
        if self._init_method == "eye":
            base = torch.eye(D, self.Q, device=device)
        else:
            M = torch.randn(D, self.Q, device=device)
            Qmat, _ = torch.linalg.qr(M, mode="reduced")
            base = Qmat  # [D,Q]

        codes_spatial = base.unsqueeze(0).repeat(L, 1, 1)          # [L,D,Q]
        x_mean_spatial = torch.zeros(L, D, device=device)          # [L,D]
        mean_count_spatial = torch.zeros(1, dtype=torch.float64, device=device)

        self.register_buffer("codes_spatial", codes_spatial)
        self.register_buffer("x_mean_spatial", x_mean_spatial)
        self.register_buffer("mean_count_spatial", mean_count_spatial)
        self.register_buffer("_update_count_spatial", torch.zeros((), dtype=torch.long, device=device))

        self._D_built_spatial = D
        self._L_built_spatial = L
        self._spatial_built.fill_(True)

    @torch.no_grad()
    def _update_mean_spatial(self, x_ld: Tensor, ema_alpha: float = 0.9):
        """
        x_ld: [L, D] batch mean of per-position channel vectors
        Updates x_mean_spatial[L,D] with EMA-by-counts (vectorized across L).
        """
        n = self.mean_count_spatial  # [L]
        S_n  = (1 - (ema_alpha ** n)) / (1 - ema_alpha)          # [L]
        S_n1 = (1 - (ema_alpha ** (n + 1))) / (1 - ema_alpha)    # [L]
        # reshape for broadcast
        S_n  = S_n.unsqueeze(-1)    # [L,1]
        S_n1 = S_n1.unsqueeze(-1)   # [L,1]
        mu_exact = (x_ld / S_n1) + (ema_alpha * S_n / S_n1) * self.x_mean_spatial
        self.x_mean_spatial.copy_(mu_exact)
        self.mean_count_spatial.add_(1)

    @torch.no_grad()
    def _codes_update_spatial(self, x_bld: Tensor, lr: float):
        """
        x_bld: [B, L, D] centered inputs
        codes_spatial: [L, D, Q]
        """
        B, L, D = x_bld.shape
        Q = self.Q

        # Y = X @ C  -> [B, L, Q]
        Y = torch.einsum('bld,ldq->blq', x_bld, self.codes_spatial)

        # Gram per location: [L,Q,Q]
        gram = torch.einsum('blq,blp->lqp', Y, Y)

        # Upper-tri mask once, apply to all locations
        # (use device + dtype of gram)
        ones = torch.ones(Q, Q, device=gram.device, dtype=gram.dtype)
        upper_mask = torch.triu(ones)
        gram_upper = gram * upper_mask  # [L,Q,Q]

        # dC = X^T Y - C @ upper(Gram)  -> [L,D,Q]
        XtY = torch.einsum('bld,blq->ldq', x_bld, Y)
        CGu = torch.einsum('ldp,lpq->ldq', self.codes_spatial, gram_upper)
        dC = (XtY - CGu) / max(1, B)
        dC_norm = torch.linalg.norm(dC, dim=(1,2), keepdim=True).clamp_min(1e-12)
        scale = (1e-1 / dC_norm).clamp_max(1.0)
        dC = dC * scale
        self.codes_spatial.add_(lr * dC)

        # periodic batched re-orthonormalization
        self._update_count_spatial.add_(1)
        if int(self._update_count_spatial.item()) % self.Q == 0:
            # C_l <- C_l (C_l^T C_l)^(-1/2) for all l
            # G: [L,Q,Q]
            G = torch.einsum('ldq,ldp->lqp', self.codes_spatial, self.codes_spatial)
            G = 0.5 * (G + G.transpose(1,2))
            normG = torch.linalg.norm(G, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1.0)
            Q = G.shape[-1]
            eye = torch.eye(Q, device=G.device).expand(*G.shape[:-2], Q, Q)
            G = G + eye * (1e-6 * normG)
            lam, U = torch.linalg.eigh(G)  # batched eigendecomp
            inv_sqrt_lam = torch.clamp(lam, min=1e-8).rsqrt()  # [L,Q]
            # S_inv = U diag(inv_sqrt_lam) U^T  -> [L,Q,Q]
            S_inv = torch.einsum('lik,lkm,ljm->lij', U, torch.diag_embed(inv_sqrt_lam), U)
            # C <- C S_inv
            self.codes_spatial.copy_(torch.einsum('ldp,lpq->ldq', self.codes_spatial, S_inv))

    # ---------- Forward ----------

    def forward(self,
                latents: Tensor,
                status: str = 'Train',
                current_lr: Optional[float] = None):
        """
        latents: [B, D, H, W]
        Returns:
          x_hat:    [B, D, H, W]
          dummy1:   scalar tensor (0.)
          dummy2:   [B, H, W] zeros (kept for API compatibility)
        """
        assert latents.dim() == 4, "Expect latents as [B, D, H, W]"
        B, D, H, W = latents.shape
        device = latents.device

        if not self.keep_shape:
            # ----- Original (classic) mode -----
            DHW = D * H * W
            x = latents.view(B, DHW)  # [B, D*H*W]

            N_target = (self.N if self._use_mapper and (self.N is not None) else DHW)
            self._ensure_codes(N_target, device)

            if self._use_mapper:
                if (self.mapper is None) or (self._in_features != DHW):
                    self._build_heads(DHW, device)
                x_N = self.mapper(x)  # [B,N]
            else:
                x_N = x  # [B,N==DHW]

            if status == 'Train':
                self._update_mean(x_N.detach())

            x_N_centered = x_N - self.xN_mean  # [B,N]

            if status == 'Train' and (current_lr is not None):
                self._codes_update(x_N_centered.detach(), current_lr)

            Y = x_N_centered @ self.codes                 # [B,Q]
            x_rec_centered = Y @ self.codes.T             # [B,N]
            x_rec_N = x_rec_centered + self.xN_mean       # [B,N]

            if self._use_mapper:
                x_hat_flat = self.decoder(x_rec_N)        # [B,DHW]
            else:
                x_hat_flat = x_rec_N

            x_hat = x_hat_flat.view(B, D, H, W)
            return x_hat, torch.tensor(0., device=device), torch.zeros((B, H, W), dtype=torch.long, device=device)

        else:
            # ----- New keep_shape mode: H*W parallel PCA over channel vectors -----
            # reshape to [B, L, D], where L=H*W and each row is a D-dim vector at (h,w)
            L = H * W
            x_bld = latents.permute(0, 2, 3, 1).contiguous().view(B, L, D)  # [B,L,D]

            self._build_spatial_if_needed(D, H, W, device)

            if status == 'Train':
                batch_mean_ld = x_bld.mean(dim=0)  # [L,D]
                self._update_mean_spatial(batch_mean_ld.detach())

            x_centered_bld = x_bld - self.x_mean_spatial.unsqueeze(0)  # [B,L,D]

            if status == 'Train' and (current_lr is not None):
                self._codes_update_spatial(x_centered_bld.detach(), current_lr)

            # Project and reconstruct per location
            # Y: [B,L,Q], x_rec_centered: [B,L,D]
            Y = torch.einsum('bld,ldq->blq', x_centered_bld, self.codes_spatial)
            x_rec_centered = torch.einsum('blq,ldq->bld', Y, self.codes_spatial)
            x_rec = x_rec_centered + self.x_mean_spatial.unsqueeze(0)  # [B,L,D]

            # back to [B,D,H,W]
            x_hat = x_rec.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

            return x_hat, torch.tensor(0., device=device), torch.zeros((B, H, W), dtype=torch.long, device=device)

    # ---------- Schedules (unchanged) ----------

    def get_decay(self, method: str, epoch: int, max_epoch: int, initial_p: float = 0.1,
                  warmup_flag: bool = False, warmup_epochs: int = 0, warmup_type: str = 'linear',
                  peak_p: float = 0.5, **kwargs) -> float:

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
                k = kwargs.get("k", 2.0)
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
                    p = peak_p * 0.5 * (1 - math.cos(math.pi * progress))
                else:
                    p = peak_p * progress
                return max(p, 1e-8)
            else:
                decay_epochs_total = max(1, max_epoch - warmup_epochs)
                decay_epoch = min(epoch - warmup_epochs, decay_epochs_total)
                p = _base_schedule(method, decay_epoch, decay_epochs_total, peak_p)
                return max(p, 1e-8)
        else:
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
                quantizerConfig: dict = None,
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
        if quantizerConfig is None:
            self.vq_layer = PCA(
                num_embeddings= num_embeddings,
                embedding_dim= None,
                method = "qr",
                )
        else:
            self.vq_layer = instantiate_from_config(quantizerConfig)
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

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
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

#%%