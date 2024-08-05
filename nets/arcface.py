import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mobilenet import MobileNetForClassification, MOBILENET_LARGE_CONFIG


class ArcFaceHead(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(x, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, weight)
    

class ArcFaceLoss(nn.Module):
    def __init__(self, margin: float = 0.5, scale: float = 1.0) -> None:
        super().__init__()
        self.s = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, cosine: Tensor, labels: Tensor) -> Tensor:
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        cosine_with_margin = cosine * self.cos_m - sine * self.sin_m

        # only positive samples have the margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(dim=1, index=labels.view(-1, 1).long(), value=1)
        
        output = (one_hot * cosine_with_margin) + ((1 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, labels) 
        
        return loss


def create_face_recognition_model(num_identities: int, embedding_dim: int) -> nn.Module:
    backbone = MobileNetForClassification(MOBILENET_LARGE_CONFIG, embedding_dim, 3)
    model = nn.Sequential(OrderedDict([
        ('backbone', backbone),
        ('head', ArcFaceHead(num_identities, embedding_dim))
    ]))
    return model
