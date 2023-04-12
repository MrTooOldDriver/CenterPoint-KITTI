from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2
from .IASSD_backbone import IASSD_Backbone
from .IASSD_backbone_DLP import IASSD_Backbone_DLP
from .IASSD_backbone_DPP import IASSD_Backbone_DPP
from .IASSD_backbone_DPPF import IASSD_Backbone_DPPF
from .IASSD_backbone_CFAR_Merge import IASSD_Backbone_CFAR_Merge
# from .multi_IASSD_backbone import multi_IASSD_Backbone
from .ThreeDSSD_backbone import SSDBackbone
from .RaDet_backbone import RaDetBackbone, RaDetBackbonev2
__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'IASSD_Backbone': IASSD_Backbone,
    'IASSD_Backbone_DLP': IASSD_Backbone_DLP,
    'IASSD_Backbone_DPP': IASSD_Backbone_DPP,
    'IASSD_Backbone_DPPF': IASSD_Backbone_DPPF,
    'IASSD_Backbone_CFAR_Merge': IASSD_Backbone_CFAR_Merge,
    # 'multi_IASSD_Backbone': multi_IASSD_Backbone,
    'SSDBackbone': SSDBackbone,
    'RaDetBackbone': RaDetBackbone,
    'RaDetBackbonev2': RaDetBackbonev2
}