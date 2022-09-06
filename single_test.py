import os
import sys
import random
import logging
import pathlib
from time import time

from PIL import Image
import numpy as np
import torch

import torchvision.transforms as transforms

from lib.models import PolyRegression

def run_inference(image_path):
    model, device = load_model()

    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    return infer(model, tensor, device)


def load_model(device: str = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model_parameters = {
        'num_outputs': 35, # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
        'pretrained': True,
        'backbone': 'resnet50',
        'pred_category': False,
        'curriculum_steps': [0, 0, 0, 0],
    }
    model = PolyRegression(**model_parameters)
    model.to(device)

    model_location = os.path.join(pathlib.Path(__file__).parent.resolve(), "model_2695_resnet50.pt")
    loaded = torch.load(model_location, map_location=device)
    model.load_state_dict(loaded['model'])

    model.eval()

    return model, device


def infer(model, tensor, device):
    with torch.no_grad():
        tensor = tensor.to(device)

        outputs = model(tensor)
        outputs = model.decode(outputs, labels=None)
        return outputs


def init():
    cfg = { "seed": 0 }

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    sys.excepthook = log_on_exception
    init()
    run_inference(sys.argv[1])
