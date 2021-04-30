# -*-coding:utf-8-*-
import torch

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def wasserstein_distance(first_samples,
                         second_samples,
                         p=2,
                         device='cuda'):
    wasserstein_distance = torch.abs(first_samples[0] - second_samples[0])
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p)), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
