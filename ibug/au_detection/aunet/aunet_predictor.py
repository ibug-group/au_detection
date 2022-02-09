import os
import torch
import numpy as np
from types import SimpleNamespace
from typing import Union, Optional, Dict
from .aunet import AUNet


__all__ = ['AUNetPredictor']


class AUNetPredictor(object):
    def __init__(self, device: Union[str, torch.device] = 'cuda:0', model: Optional[SimpleNamespace] = None,
                 config: Optional[SimpleNamespace] = None) -> None:
        self.device = device
        if model is None:
            model = AUNetPredictor.get_model()
        if config is None:
            config = AUNetPredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = AUNet(config=self.config).to(self.device)
        self.net.load_state_dict(torch.load(model.weights, map_location=self.device))
        self.net.eval()
        if self.config.use_jit:
            self.net = torch.jit.trace(self.net, torch.rand(
                1, self.config.num_input_channels, self.config.input_size, self.config.input_size).to(self.device))

    @staticmethod
    def create_config(use_jit: bool = True) -> SimpleNamespace:
        return SimpleNamespace(use_jit=use_jit)

    @staticmethod
    def get_model(name: str = 'aunet_bdaw') -> SimpleNamespace:
        name = name.lower()
        if name == 'aunet_bdaw':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'aunet_bdaw.pth'),
                                   config=SimpleNamespace(
                                       num_input_channels=768, input_size=64, n_blocks=4,
                                       au_indices=(1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24, 25, 26)))
        elif name == 'aunet_bdaw2':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'aunet_bdaw2.pth'),
                                   config=SimpleNamespace(
                                       num_input_channels=768, input_size=64, n_blocks=4,
                                       au_indices=(1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24, 25, 26)))
        elif name == 'aunet_bdaw_vae':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'aunet_bdaw_vae.pth'),
                                   config=SimpleNamespace(
                                       num_input_channels=768, input_size=64, n_blocks=4,
                                       au_indices=(1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24, 25, 26)))
        else:
            raise ValueError("name must be set to either aunet_bdaw, aunet_bdaw2, or aunet_bdaw_vae")

    @torch.no_grad()
    def __call__(self, fan_features: torch.Tensor) -> np.ndarray:
        if fan_features.numel() > 0:
            results = self.net(fan_features.to(self.device))
            results = torch.sigmoid(results).cpu().numpy()
            return results
        else:
            return np.empty(shape=(0, len(self.config.au_indices)), dtype=np.float32)
