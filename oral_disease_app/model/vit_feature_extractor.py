import torch
import torch.nn as nn
import timm

class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)
