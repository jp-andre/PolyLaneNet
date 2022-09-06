import os
import sys
import logging
import pathlib

from PIL import Image
import torch

import torchvision.transforms as transforms

from lib.models import PolyRegression

def run_inference(image_path):
    model, device = load_model()

    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    outputs = infer(model, tensor, device)
    return outputs


def load_model(device: str = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # FIXME these two go together
    # model_parameters = {
    #     'num_outputs': 35, # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
    #     'pretrained': True,
    #     'backbone': 'resnet50',
    #     'pred_category': False,
    #     'curriculum_steps': [0, 0, 0, 0],
    # }
    # model_weights = 'model_tusimple_resnet50_2695.pt'

    model_parameters = {
        'num_outputs': 35, # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
        'pretrained': True,
        'backbone': 'efficientnet-b0',
        'pred_category': False,
        'curriculum_steps': [0, 0, 0, 0],
    }
    model_weights = 'model_tusimple_2695.pt'

    model = PolyRegression(**model_parameters)
    model.to(device)

    model_location = os.path.join(pathlib.Path(__file__).parent.resolve(), model_weights)
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


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    sys.excepthook = log_on_exception
    run_inference(sys.argv[1])
