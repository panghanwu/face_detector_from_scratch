import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mobilenet import MOBILENET_LARGE_CONFIG, MobileNetForClassification


class ArcFaceHead(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(x, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, weight)
    

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, margin: float = 0.1, scale: float = 1.0) -> None:
        super().__init__()
        self.s = scale
        self.n_classes = num_classes
        self.margin = margin
        self.upper = math.cos(margin)
        self.lower = math.cos(math.pi - margin)

    def forward(self, cosine: Tensor, labels: Tensor) -> Tensor:
        positive = F.one_hot(labels, self.n_classes) == 1
        in_range = torch.logical_and(cosine > self.lower, cosine < self.upper)
        use_margin = torch.logical_and(positive, in_range)

        theta = torch.arccos(torch.clamp(cosine[use_margin], -1+1e-7, 1-1e-7))
        cosine[use_margin] = torch.cos(theta + self.margin)
        
        loss = F.cross_entropy(self.s * cosine, labels) 
        return loss


def create_face_recognition_model(num_identities: int, embedding_dim: int, dropout: float = 0.0) -> nn.Module:
    backbone = MobileNetForClassification(MOBILENET_LARGE_CONFIG, embedding_dim, 3, dropout=dropout)
    model = nn.Sequential(OrderedDict([
        ('backbone', backbone),
        ('head', ArcFaceHead(num_identities, embedding_dim))
    ]))
    return model
