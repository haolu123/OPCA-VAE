import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *                  # import from model/types_.py
import math

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 alpha_q: float = 0.5,
                 random_prob: float = 0.0,
                 decay_method: str = "constant",
                 decay_kwargs: dict = None,
                 soft_sample: bool = False): # default p = 0
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        self.alpha = alpha_q
        self.random_prob = random_prob  # initial probability
        self.decay_method = decay_method
        self.decay_kwargs = decay_kwargs or {}  # hyperparameters for decay

        self.soft_sample_flag = soft_sample

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
            encoding_inds = self.soft_sample(dist, tau=tau)
        else:
            encoding_inds = torch.argmin(dist, dim=1)  # [BHW, 1]      


        # Compute probability p with decay schedule
        p = self.random_prob
        if epoch is not None and max_epoch is not None:
            p = self.get_decay_p(
                method=self.decay_method,
                epoch=epoch,
                max_epoch=max_epoch,
                initial_p=self.random_prob,
                **self.decay_kwargs
            )
        else:
            p = 0

        # Random sampling mask
        if p > 0 and status == 'Train':
            mask = (torch.rand_like(encoding_inds.float()) < p)
            random_inds = torch.randint(0, self.K, encoding_inds.shape, device=latents.device)
            encoding_inds = torch.where(mask, random_inds, encoding_inds)

        encoding_inds = encoding_inds.unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        if self.alpha > 0: 
            vq_loss = commitment_loss * self.beta + embedding_loss * self.alpha
            
            alpha_q = 2 * self.alpha / (latents_shape[0] * latents_shape[1] * latents_shape[2] * latents_shape[3])
            quantized_latents = latents + alpha_q * quantized_latents + ((1-alpha_q)*quantized_latents - latents).detach() # method 3, corrected formula
        else:
            vq_loss = commitment_loss * self.beta + embedding_loss
             # Add the residue back to the latents
            quantized_latents = latents + (quantized_latents - latents).detach()         # standard method 0
            # quantized_latents = latents + quantized_latents - latents.detach()             # method 1
            # quantized_latents = latents + self.alpha * ( quantized_latents - latents).detach()  # method 2
            
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
        probs = F.softmax(logits, dim=1)         # [B, K]

        # 按概率分布采样一个索引
        indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

        return indices

    def get_decay_p(self, method: str, epoch: int, max_epoch: int, initial_p: float = 0.1, **kwargs) -> float:
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
        
        Returns
        -------
        float
            Probability p (>=0)
        """
        
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
        
        return max(p, 0.0)



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
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 alpha: float = 0.5,
                 random_prob: float = 0.0,
                 img_size: int = 64,
                 decay_method: str = "constant",          # New
                 decay_kwargs: dict = None,             # New
                 soft_sample: bool = False,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

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
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=self.beta,
            alpha_q=self.alpha,
            random_prob=random_prob,
            decay_method=decay_method,      # pass decay method
            decay_kwargs=decay_kwargs,       # pass decay hyperparameters
            soft_sample = soft_sample
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