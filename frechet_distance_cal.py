import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as opt
import numpy as np
import torchvision
import torch.nn.functional as func
import torchvision.transforms as tf_transforms
from scipy import linalg
from torchvision import models

compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelInceptionV3(nn.Module):
    BLOCK_IDX_DEFAULT = 3
    DIM_TO_INDEX = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3
    }

    def __init__(self,
                 blocks_output=[BLOCK_IDX_DEFAULT],
                 input_resize=True,
                 input_normalize=True,
                 grad_required=False):
        
        super(ModelInceptionV3, self).__init__()

        self.input_resize = input_resize
        self.input_normalize = input_normalize
        self.blocks_output = sorted(blocks_output)
        self.block_final = max(blocks_output)

        assert self.block_final <= 3, \
            'Index of output block can be at most 3'

        self.model_blocks = nn.ModuleList()
        model_inception = models.inception_v3(pretrained=True)

        # Block A: from input to maxpool1
        block_a = [
            model_inception.Conv2d_1a_3x3,
            model_inception.Conv2d_2a_3x3,
            model_inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.model_blocks.append(nn.Sequential(*block_a))

        # Block B: from maxpool1 to maxpool2
        if self.block_final >= 1:
            block_b = [
                model_inception.Conv2d_3b_1x1,
                model_inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.model_blocks.append(nn.Sequential(*block_b))

        # Block C: from maxpool2 to auxiliary classifier
        if self.block_final >= 2:
            block_c = [
                model_inception.Mixed_5b,
                model_inception.Mixed_5c,
                model_inception.Mixed_5d,
                model_inception.Mixed_6a,
                model_inception.Mixed_6b,
                model_inception.Mixed_6c,
                model_inception.Mixed_6d,
                model_inception.Mixed_6e,
            ]
            self.model_blocks.append(nn.Sequential(*block_c))

        # Block D: from auxiliary classifier to final average pooling
        if self.block_final >= 3:
            block_d = [
                model_inception.Mixed_7a,
                model_inception.Mixed_7b,
                model_inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.model_blocks.append(nn.Sequential(*block_d))

        for param in self.parameters():
            param.requires_grad = grad_required

    def forward(self, input_tensor):
        output_tensor = []
        x = input_tensor

        if self.input_resize:
            x = func.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.input_normalize:
            x = 2 * x - 1  # Normalize from (0, 1) to (-1, 1)

        for idx, block in enumerate(self.model_blocks):
            x = block(x)
            if idx in self.blocks_output:
                output_tensor.append(x)

            if idx == self.block_final:
                break

        return output_tensor
    


def compute_stats_activation(images, inception_model, batch_sz=128, dimensions=2048, use_cuda=False):
    inception_model.eval()
    activations=np.empty((len(images), dimensions))
    
    if use_cuda:
        batch=images.cuda()
    else:
        batch=images
    prediction = inception_model(batch)[0]
    if prediction.size(2) != 1 or prediction.size(3) != 1:
        prediction = func.adaptive_avg_pool2d(prediction, output_size=(1, 1))

    activations= prediction.cpu().data.numpy().reshape(prediction.size(0), -1)
    
    mean_act = np.mean(activations, axis=0)
    covariance_act = np.cov(activations, rowvar=False)
    return mean_act, covariance_act

def compute_distance_frechet(mu1, sigma1, mu2, sigma2, epsilon=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Means of training and test are different lengths'
    assert sigma1.shape == sigma2.shape, 'Dimensions of training and test covariances differ'

    diff_mu = mu1 - mu2

    sigma_product_sqrt, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(sigma_product_sqrt).all():
        errmsg = ('FID calculation gives singular product; '
               'adding %s to diagonal of covariance estimates') % epsilon
        print(errmsg)
        offset_identity = np.eye(sigma1.shape[0]) * epsilon
        sigma_product_sqrt = linalg.sqrtm((sigma1 + offset_identity).dot(sigma2 + offset_identity))

    if np.iscomplexobj(sigma_product_sqrt):
        if not np.allclose(np.diagonal(sigma_product_sqrt).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(sigma_product_sqrt.imag))
            raise ValueError(f'Imaginary component {max_imag}')
        sigma_product_sqrt = sigma_product_sqrt.real

    trace_sigma_product_sqrt = np.trace(sigma_product_sqrt)

    return (diff_mu.dot(diff_mu) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * trace_sigma_product_sqrt)

def calculate_fretchet(real_images, fake_images):
    inception_block_idx = ModelInceptionV3.DIM_TO_INDEX[2048]
    inception_model = ModelInceptionV3([inception_block_idx])
    inception_model = inception_model.cuda()
    real_mu, real_sigma = compute_stats_activation(real_images, inception_model, use_cuda=True)
    fake_mu, fake_sigma = compute_stats_activation(fake_images, inception_model, use_cuda=True)
    fid_score = compute_distance_frechet(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_score
