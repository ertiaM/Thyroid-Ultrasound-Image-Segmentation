# encoding: utf-8
from .vgg import VGG16


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.NAME == 'vgg16':
        backbone = VGG16()
        return backbone
