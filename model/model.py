""" 1. Input: n X m """
from .models.LeNet5 import LeNet5

from .models.DenseNet import DenseNet121, DenseNet161, DenseNet169, DenseNet201

from .models.ConvNeXt_v2 import convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge

from .models.InceptionNeXt import inceptionnext_tiny, inceptionnext_small, inceptionnext_base, inceptionnext_base_384

from .models.VanillaNet import vanillanet_5, vanillanet_6, vanillanet_7, vanillanet_8, vanillanet_9, vanillanet_10, vanillanet_11, vanillanet_12
from .models.VanillaNet import vanillanet_13, vanillanet_13_x1_5, vanillanet_13_x1_5_ada_pool

from .models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

""" 2. Input: n X n -> need to pad for model """
from .models.DeiT3 import deit_tiny_patch16_LS, deit_small_patch16_LS, deit_base_patch16_LS
from .models.DeiT3 import deit_large_patch16_LS, deit_huge_patch14_LS

from .models.HiFuse import HiFuse_Tiny, HiFuse_Small, HiFuse_Base

from .models.ResT_v2 import restv2_tiny, restv2_small, restv2_base, restv2_large
