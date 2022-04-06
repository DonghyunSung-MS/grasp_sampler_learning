from .flow import FlowGraspNet
from .vae import VAEGraspNet
from .gan import GANGraspNet

MODEL_ZOO = {
    "FLOW": FlowGraspNet,
    "VAE": VAEGraspNet,
    "GAN": GANGraspNet
}
