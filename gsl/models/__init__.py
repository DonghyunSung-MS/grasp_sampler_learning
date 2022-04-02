from .flow import FlowGraspNet
from .vae import VAEGraspNet

MODEL_ZOO = {
    "FLOW": FlowGraspNet,
    "VAE": VAEGraspNet,
}
