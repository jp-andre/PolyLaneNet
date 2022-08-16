import os
import sys
import random
import logging
from time import time

import cv2
import numpy as np
import torch

import torchvision.transforms as transforms

from lib.models import PolyRegression

def run_inference(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_parameters = {
    "num_outputs": 35, # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
    "pretrained": True,
    "backbone": 'efficientnet-b0',
    "pred_category": False,
    "curriculum_steps": [0, 0, 0, 0]
    }
    model = PolyRegression(**model_parameters)
    model.to(device)

    loaded = torch.load("model.pt")
    model.load_state_dict(loaded['model'])
    model.eval()

    return infer(model, image_path, device)


def infer(model, image_path: str, device = "cpu"):

    transform = transforms.ToTensor()

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    print("image shape", image.shape)
    print("tensor shape", tensor.shape)

    with torch.no_grad():
        tensor = tensor.to(device)

        t0 = time()
        outputs = model(tensor)
        t = time() - t0
        print("Inference time", t)

        print("Outputs:", outputs)
        outputs = model.decode(outputs, labels=None)
        print("Decoded:", outputs)


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def main(path: str):
    cfg = { "seed": 0 }

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    exp_root = "workdir"
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

    run_inference(path)


if __name__ == "__main__":
    sys.excepthook = log_on_exception
    main(sys.argv[1])
