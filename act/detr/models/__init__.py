from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp
from .transformer import build_transformer
from .backbone import build_backbone
from .position_encoding import build_position_encoding

def build_ACT_model(args):
    return build_vae(args)


build_me_ACT_model = build_ACT_model

def build_CNNMLP_model(args):
    return build_cnnmlp(args)
