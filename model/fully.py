#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import UNet as MonaiUNet

from monai.networks.nets import DynUNet 

from monai.networks.nets import BasicUNet as MonaiBasicUNet



class FullLearn(nn.Module):
    """
    Autoencoder with:
      - A single-channel latent in [0, 1].
      - An additional alpha input with alpha_in_channels.
      - The decoder sees (latent + alpha_input) => total dec_in_channels.

    forward(x, alpha_x) -> (latent, reconstructed)
    """
    def __init__(
        self,
        settings,
        final_activation,
        in_channels: int,
        out_channels: int = None, # typically same as in_channels if reconstructing
    ):
        super(FullLearn, self).__init__()


        def linear (x):
            return x
        def clamp01(x):
            return torch.clamp(x, 0, 1)

        if final_activation == "sigmoid":
            self.final_activation = torch.sigmoid
        elif final_activation == "clamp":
            self.final_activation = clamp01
        elif final_activation == "relu":
            self.final_activation = F.relu
        elif final_activation == "linear":
            self.final_activation = linear
        elif final_activation == "tanh":
            self.final_activation = torch.tanh 
        else:
            raise ValueError("final_activation not found")

        if out_channels is None:
            out_channels = in_channels


        if settings["name"]  == "DynUNet":
            self.model = DynUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=settings["kernel_size"],
                strides=settings["strides"],
                upsample_kernel_size = settings["upsample_kernel_size"],
                res_block=settings["res_block"],
                norm_name=settings["norm_name"]
                
            )
        else:
            raise ValueError("Decoder name not found")


    def forward(self, x, distanceMap, brainMask):
        """
        x       : (N, in_channels, D, H, W)
        alpha_x : (N, alpha_in_channels, D, H, W)
                  must match the same spatial size as the latent
        Returns:
           latent: (N, 1, D, H, W) in [0,1]
           output: (N, out_channels, D, H, W)
        """

        x = self.model(x)

        # this is between -1 and  1 
        x = torch.clamp( self.final_activation(x) * brainMask, -1, 1)

        x = x + distanceMap 

        x = torch.clamp(x, 0, 1)


        return x


if __name__ == "__main__":
    model = FullLearn(
        settings = {
            "name": "DynUNet",
            "kernel_size": (3,3,3,3,3),
            "upsample_kernel_size": (2, 2, 2,2),
            "strides": (1, 2, 2, 2,2),
            "res_block" : False,
        },
        final_activation = "sigmoid",
        in_channels = 1
    )

    x = torch.randn(1, 1, 64, 64, 64)
    distanceMap = torch.randn(1, 1, 64, 64, 64)
    brainMask = torch.randn(1, 1, 64, 64, 64)

    x = model(x, distanceMap, brainMask)
    print(x.shape)



# %%
