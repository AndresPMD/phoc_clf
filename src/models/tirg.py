import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ConCatModule(nn.Module):

  def __init__(self):
    super(ConCatModule, self).__init__()

  def forward(self, x):
    x = torch.cat(x, dim=1)
    return x

class TIRG(nn.Module):
  """The TIGR model.
  The method is described in
  Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
  "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
  CVPR 2019. arXiv:1812.07119
  """

  def __init__(self, img_input_dim, text_input_dim, output_dim):
    super(TIRG, self).__init__()

    self.a = nn.Parameter(torch.tensor([1.0, 0.1]))
    self.gated_feature_composer = torch.nn.Sequential(
        ConCatModule(), nn.BatchNorm1d(img_input_dim + text_input_dim), torch.nn.ReLU(),
        torch.nn.Linear(img_input_dim + text_input_dim, output_dim))
    self.res_info_composer = torch.nn.Sequential(
        ConCatModule(), torch.nn.BatchNorm1d(img_input_dim + text_input_dim), torch.nn.ReLU(),
        torch.nn.Linear(img_input_dim + text_input_dim, img_input_dim + text_input_dim), torch.nn.ReLU(),
        torch.nn.Linear(img_input_dim + text_input_dim, output_dim))

  def forward(self, x):
    # input format: x = [img_features, text_features]
    img_features = x[0]
    fgate = self.gated_feature_composer(x)
    fres  = self.res_info_composer(x)
    f = F.sigmoid(fgate) * img_features * self.a[0] + fres * self.a[1]
    return f
