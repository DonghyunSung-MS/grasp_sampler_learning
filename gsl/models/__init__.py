from .flow import FlowGraspNet
from .gan import GANGraspNet
from .vae import VAEGraspNet

MODEL_ZOO = {"FLOW": FlowGraspNet, "VAE": VAEGraspNet, "GAN": GANGraspNet}
