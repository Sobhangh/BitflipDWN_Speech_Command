import torch
import torch.nn as nn
from torchlogix.layers import OrPooling2d, GroupSum, LogicConv2d, LogicDense
from torchlogix.layers.binarization import setup_binarization

class ClgnO3(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 2-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """
    n_input_bits = None
    k_num = None
    tau = None
    group_size = None
    group_size_input = None

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, connections_kwargs: dict, **llkw):
        assert thresholds.shape[-1] == self.n_input_bits, f"{self.__class__.__name__} model requires {self.n_input_bits}-bit thresholds."
        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_bits = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        connections_kwargs = dict(connections_kwargs)  # make a copy to avoid modifying the original
        connections_kwargs["channel_group_size"] = self.group_size  # from the paper, we use grouped connections with channel group size 2 for conv layers

        group_size_input = self.group_size_input if self.group_size_input is not None else self.group_size
        
        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=64,
                num_kernels=self.k_num,
                channels=3*n_bits,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs | {"channel_group_size": group_size_input},
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # kx32x32

        layers.append(
            LogicConv2d(
                in_dim=32,
                channels=self.k_num,
                num_kernels=4*self.k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 4kx16x16

        layers.append(
            LogicConv2d(
                in_dim=16,
                channels=4*self.k_num,
                num_kernels=16*self.k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 16kx8x8
        
        layers.append(
            LogicConv2d(
                in_dim=8,
                channels=16*self.k_num,
                num_kernels=32*self.k_num,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=connections_kwargs,
                **llkw,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 32kx4x4

        layers.append(
                LogicConv2d(
                    in_dim=4,
                    channels=32*self.k_num,
                    num_kernels=64*self.k_num,
                    tree_depth=2,
                    receptive_field_size=2,
                    padding=1,
                    connections_kwargs=connections_kwargs,
                    **llkw,
                )
                )
        layers.append(OrPooling2d(kernel_size=2, stride=2)) # 64kx2x2

        layers.append(torch.nn.Flatten()) # 256k

        layers.append(LogicDense(in_dim=256*self.k_num, out_dim=1280*self.k_num, **llkw))
        layers.append(LogicDense(in_dim=1280*self.k_num, out_dim=640*self.k_num, **llkw))
        layers.append(LogicDense(in_dim=640*self.k_num, out_dim=320*self.k_num, **llkw))

        super(ClgnO3, self).__init__(*layers, GroupSum(k=10, tau=self.tau))

