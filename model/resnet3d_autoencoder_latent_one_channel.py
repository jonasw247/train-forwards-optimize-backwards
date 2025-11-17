#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import UNet as MonaiUNet

from monai.networks.nets import DynUNet 

from monai.networks.nets import BasicUNet as MonaiBasicUNet

class BasicBlock3D(nn.Module):
    """
    A basic 3D residual block:
    - Convolution -> BatchNorm -> ReLU -> Convolution -> BatchNorm
    - Skip connection (identity or 1x1 conv if channels change)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock3D, self).__init__()
        padding = kernel_size // 2  # keeps spatial size the same

        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Optional skip connection if in/out channels differ
        self.skip_connection = None
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    """
    3D Encoder that ends in exactly 1 latent channel in [0,1].
    Internally uses enc_hidden_channels, then outputs latent_channels=1.

    - input:  (N, in_channels, D, H, W)
    - output: (N, 1, D, H, W)  [latent in 0..1]
    """
    def __init__(
        self,
        in_channels: int,
        enc_hidden_channels: int,
        enc_num_blocks: int,
        enc_kernel_size: int = 3
    ):
        super(ResNet3DEncoder, self).__init__()

        # --------------------------------------------------
        # Initial conv: in_channels -> enc_hidden_channels
        # --------------------------------------------------
        self.conv_in = nn.Conv3d(
            in_channels, enc_hidden_channels,
            kernel_size=enc_kernel_size,
            stride=1,
            padding=enc_kernel_size // 2,
            bias=False
        )
        self.bn_in = nn.BatchNorm3d(enc_hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # --------------------------------------------------
        # Stack of residual blocks
        # --------------------------------------------------
        blocks = []
        for _ in range(enc_num_blocks):
            blocks.append(
                BasicBlock3D(
                    in_channels=enc_hidden_channels,
                    out_channels=enc_hidden_channels,
                    kernel_size=enc_kernel_size
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # --------------------------------------------------
        # Final conv to collapse to 1 channel (the latent)
        # --------------------------------------------------
        self.conv_latent = nn.Conv3d(
            enc_hidden_channels, 1,  # latent_channels = 1
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        # (N, in_channels, D, H, W)
        x = self.conv_in(x)     # -> (N, enc_hidden_channels, D, H, W)
        x = self.bn_in(x)
        x = self.relu(x)

        x = self.blocks(x)      # -> (N, enc_hidden_channels, D, H, W)

        # Map to 1 channel
        x = self.conv_latent(x) # -> (N, 1, D, H, W)

        # Constrain to [0, 1]
        #x = self.relu(x)
        #x = torch.clamp(x, 0, 1)
        #x = torch.sigmoid(x)
        return x


class ResNet3DDecoder(nn.Module):
    """
    3D Decoder:
      input:  (N, dec_in_channels, D, H, W)
      output: (N, out_channels, D, H, W)

    dec_in_channels = (1 + alpha_in_channels) if the latent is 1 channel
    """
    def __init__(
        self,
        dec_in_channels: int,
        out_channels: int,
        dec_hidden_channels: int,
        dec_num_blocks: int,
        dec_kernel_size: int = 3
    ):
        super(ResNet3DDecoder, self).__init__()

        # --------------------------------------------------
        # Initial conv: dec_in_channels -> dec_hidden_channels
        # --------------------------------------------------
        self.conv_in = nn.Conv3d(
            dec_in_channels, dec_hidden_channels,
            kernel_size=dec_kernel_size,
            stride=1,
            padding=dec_kernel_size // 2,
            bias=False
        )
        self.bn_in = nn.BatchNorm3d(dec_hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        blocks = []
        for _ in range(dec_num_blocks):
            blocks.append(
                BasicBlock3D(
                    in_channels=dec_hidden_channels,
                    out_channels=dec_hidden_channels,
                    kernel_size=dec_kernel_size
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.conv_out = nn.Conv3d(
            dec_hidden_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        # (N, dec_in_channels, D, H, W)
        x = self.conv_in(x)     # -> (N, dec_hidden_channels, D, H, W)
        x = self.bn_in(x)
        x = self.relu(x)

        x = self.blocks(x)      # -> (N, dec_hidden_channels, D, H, W)
        x = self.conv_out(x)    # -> (N, out_channels, D, H, W)
        return x

#############################################
# Additional Diffusion Decoder for 3D MRI Images
#############################################

class DiffusionDecoder3D(nn.Module):
    """
    A lightweight, differentiable DDIM-style diffusion decoder.
    
    It conditions on the concatenated latent and alpha input (the "conditioning")
    and iteratively refines an initial guess. Because the sampling loop is
    deterministic (eta=0) and built from differentiable operations, gradients
    flow back to the conditioning (and thus to the latent and encoder).
    
    Parameters:
      - in_channels: number of channels in the conditioning input (latent + alpha)
      - out_channels: number of output image channels
      - hidden_channels: number of channels in the internal (denoising) network
      - num_steps: number of diffusion (denoising) steps (keep small, e.g. 4)
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_steps):
        super(DiffusionDecoder3D, self).__init__()
        self.num_steps = num_steps
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # The denoising network (a small 3D conv net)
        # It takes as input the concatenation of the current state x (with out_channels)
        # and the conditioning (with in_channels) and predicts a noise residual.
        self.diffusion_net = nn.Sequential(
            nn.Conv3d(in_channels + out_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)

        # A small time embedding network (for simple time conditioning)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # A simple initialization: map the conditioning to an initial x_T.
        self.init_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Precompute a simple linear diffusion schedule (betas linearly spaced)
        betas = torch.linspace(1e-4, 0.02, steps=num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def forward_diffusion(self, x, t, conditioning):
        """
        One denoising step:
          - x: current state tensor of shape (N, out_channels, D, H, W)
          - t: an integer time step (from 0 to num_steps-1)
          - conditioning: conditioning tensor of shape (N, in_channels, D, H, W)
        Returns:
          - noise prediction of shape (N, out_channels, D, H, W)
        """
        # Create a (scalar) time tensor and get its embedding.
        t_tensor = torch.tensor([t], device=x.device, dtype=torch.float32).unsqueeze(0)  # shape (1, 1)
        t_emb = self.time_embed(t_tensor)  # shape (1, hidden_channels)
        t_emb = t_emb.view(x.shape[0], self.hidden_channels, 1, 1, 1)  # expand to (N, hidden_channels, 1,1,1)
        
        # Concatenate the current state and conditioning along the channel dim.
        x_input = torch.cat([x, conditioning], dim=1)  # shape: (N, in_channels+out_channels, D, H, W)
        h = self.diffusion_net(x_input)
        # Add in the (broadcasted) time embedding.
        h = h + t_emb
        noise_pred = self.final_conv(h)
        return noise_pred

    def forward(self, conditioning):
        """
        Run a deterministic DDIM-style sampling loop.
        
        Args:
          - conditioning: (N, in_channels, D, H, W) -- the concatenated latent and alpha.
        Returns:
          - output: (N, out_channels, D, H, W) -- the reconstructed image.
        """
        # Initialize x_T from the conditioning (learned initialization)
        x = self.init_conv(conditioning)  # (N, out_channels, D, H, W)
        
        # Iteratively denoise from t = num_steps-1 down to 0.
        for t in reversed(range(self.num_steps)):
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            # Predict noise residual at this step.
            noise_pred = self.forward_diffusion(x, t, conditioning)
            # Estimate x0 (the “clean” image) from the current x and noise prediction.
            x0_pred = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            if t > 0:
                sqrt_alpha_prev = self.sqrt_alphas_cumprod[t - 1]
                sqrt_one_minus_alpha_prev = self.sqrt_one_minus_alphas_cumprod[t - 1]
                # DDIM update rule (deterministic, with eta=0)
                x = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * noise_pred
            else:
                x = x0_pred
        return x


class ResNet3DAutoencoder(nn.Module):
    """
    Autoencoder with:
      - A single-channel latent in [0, 1].
      - An additional alpha input with alpha_in_channels.
      - The decoder sees (latent + alpha_input) => total dec_in_channels.

    forward(x, alpha_x) -> (latent, reconstructed)
    """
    def __init__(
        self,
        # Encoder config
        in_channels: int,
        enc_hidden_channels: int,
        enc_num_blocks: int,
        enc_kernel_size: int,

        # Additional alpha input
        alpha_in_channels: int,

        # Decoder config
        dec_hidden_channels: int,
        dec_num_blocks: int,
        dec_kernel_size: int,
        encoderSettings,
        decoderSettings,
        latent_activation,
        final_activation,
        use_extra_conv: bool = False,

        out_channels: int = None, # typically same as in_channels if reconstructing
    ):
        super(ResNet3DAutoencoder, self).__init__()

        if use_extra_conv:
            if use_extra_conv == True: # check if int
                use_extra_conv = 3
            self.extra_conv = nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=use_extra_conv,
                stride=1,
                padding = use_extra_conv // 2 
            )

        def clamp01(x):
            return torch.clamp(x, 0, 1)

        if latent_activation == "sigmoid":
            self.latent_activation = torch.sigmoid
        elif latent_activation == "clamp":
            self.latent_activation = clamp01

        if final_activation == "sigmoid":
            self.final_activation = torch.sigmoid
        elif final_activation == "clamp":
            self.final_activation = clamp01
        elif final_activation == "relu":
            self.final_activation = F.relu

        if out_channels is None:
            out_channels = in_channels


        if encoderSettings["name"] == "ResNet3DEncoder":
            self.encoder = ResNet3DEncoder(
                in_channels=in_channels,
                enc_hidden_channels=enc_hidden_channels,
                enc_num_blocks=enc_num_blocks,
                enc_kernel_size=enc_kernel_size
            )
        elif encoderSettings["name"] == "DynUNet":
            self.encoder = DynUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=1,
                kernel_size=encoderSettings["kernel_size"],
                strides=encoderSettings["strides"],
                upsample_kernel_size = encoderSettings["upsample_kernel_size"],
                res_block=encoderSettings["res_block"]
            )

        elif encoderSettings["name"] == "UNet":
            self.encoder = MonaiUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=1,
                channels= encoderSettings["channels"],
                strides= encoderSettings["strides"],
                num_res_units = encoderSettings["num_res_units"])
            
        elif encoderSettings["name"] == "BasicUNet":
            self.encoder = MonaiBasicUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=1,
                features= decoderSettings["channels"])
                #channels= encoderSettings["channels"],
                #strides= encoderSettings["strides"])
            
        else:
            raise ValueError("Encoder name not found")

        # The decoder input channels = (latent_channels=1) + alpha_in_channels
        dec_in_channels = 1 + alpha_in_channels

        if decoderSettings["name"] == "ResNet3DDecoder":
            self.decoder = ResNet3DDecoder(
                dec_in_channels=dec_in_channels,
                out_channels=out_channels,
                dec_hidden_channels=dec_hidden_channels,
                dec_num_blocks=dec_num_blocks,
                dec_kernel_size=dec_kernel_size
            )

        elif decoderSettings["name"] == "DynUNet":
            self.decoder = DynUNet(
                spatial_dims=3,
                in_channels=dec_in_channels,
                out_channels=out_channels,
                kernel_size=decoderSettings["kernel_size"],
                strides=decoderSettings["strides"],
                upsample_kernel_size = decoderSettings["upsample_kernel_size"],
                res_block=encoderSettings["res_block"]
                
            )

        elif decoderSettings["name"] == "UNet":
            self.decoder = MonaiUNet(
                spatial_dims=3,
                in_channels=dec_in_channels,
                out_channels=out_channels,
                channels= decoderSettings["channels"],
                strides= decoderSettings["strides"],
                num_res_units = decoderSettings["num_res_units"])
            
        elif decoderSettings["name"] == "BasicUNet":            
            self.decoder = MonaiBasicUNet(
                spatial_dims=3,
                in_channels=dec_in_channels,
                out_channels=out_channels,
                features= decoderSettings["channels"])
                #channels= decoderSettings["channels"],
                #strides= decoderSettings["strides"])
        
        elif decoderSettings["name"] == "DiffusionDecoder":
            self.decoder = DiffusionDecoder3D(
                in_channels=dec_in_channels,  # = latent (1 channel) + alpha_in_channels
                out_channels=out_channels,
                hidden_channels=decoderSettings.get("hidden_channels", 32),
                num_steps=decoderSettings.get("num_steps", 4)
            )

        else:
            raise ValueError("Decoder name not found")


    def forward(self, x, alpha_x, brainMask):
        """
        x       : (N, in_channels, D, H, W)
        alpha_x : (N, alpha_in_channels, D, H, W)
                  must match the same spatial size as the latent
        Returns:
           latent: (N, 1, D, H, W) in [0,1]
           output: (N, out_channels, D, H, W)
        """
        feat_map = self.encoder(x)
        latent = self.latent_activation(feat_map)

        #restrict the latent to the mask
        latent = latent * brainMask
        #feat_map = feat_map * brainMask

        combined = torch.cat([latent, alpha_x], dim=1)
        reconstructed = self.decoder(combined)
        
        if hasattr(self, "extra_conv"):
            reconstructed = self.extra_conv(reconstructed)
        
        reconstructed = self.final_activation(reconstructed)

        #set outside of mask to zero
        reconstructed = reconstructed * brainMask
       

        return latent, reconstructed


if __name__ == "__main__":
    # Example usage
    #
    # Suppose:
    #  - Input has 2 channels
    #  - alpha_in_channels = 3
    #  - Encoder: hidden_channels=16, num_blocks=4, kernel_size=3
    #  - Decoder: hidden_channels=8, num_blocks=2, kernel_size=3
    #  - We want to reconstruct 2 channels (same as the input).
    #
    # The encoder's final output is 1 channel in [0,1].
    # The decoder sees that 1 channel + alpha_in_channels=3 => total 4 channels.

    model = ResNet3DAutoencoder(
        # Encoder config
        in_channels=2,
        enc_hidden_channels=16,
        enc_num_blocks=4,
        enc_kernel_size=3,

        # Additional alpha input
        alpha_in_channels=3,

        # Decoder config
        dec_hidden_channels=8,
        dec_num_blocks=2,
        dec_kernel_size=3,
        out_channels=2
    )

    # Dummy inputs
    main_input = torch.randn(1, 2, 32, 32, 32)    # (batch=1, 2 channels)
    alpha_input = torch.randn(1, 3, 32, 32, 32)   # (batch=1, 3 channels)

    latent, output = model(main_input, alpha_input)

    print("Latent shape :", latent.shape)   # (1, 1, 32, 32, 32)
    print("Output shape :", output.shape)   # (1, 2, 32, 32, 32)
    print("Latent range : [{:.3f}, {:.3f}]".format(latent.min().item(), latent.max().item()))

# %%
class DoubleConv3D(nn.Module):
    """
    A common building block:
    - 2 consecutive 3D convolutions (kernel_size=3, padding=1)
    - Batch normalization
    - ReLU activation
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


