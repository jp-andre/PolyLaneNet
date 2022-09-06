import os
import sys
import logging
import pathlib
import numpy as np
import similaritymeasures

import cv2
import torch

import torchvision.transforms as transforms

from lib.models import PolyRegression


def load_image(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(img_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
    return image


def run_inference(image):
    model, device = load_model()
    transform = transforms.ToTensor()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    outputs = infer(model, tensor, device)
    return outputs


def fetch_model_weights(name):
    url = 'https://collimator-devops-resources.s3.us-west-2.amazonaws.com/ml-demos/PolyLaneNet/' + name
    homedir = os.environ['HOME']
    cachefile = f"{homedir}/.cache/{name}"
    if not os.path.exists(cachefile):
        os.system(f"curl -o {cachefile} {url}")
    os.system(f"ln -sf {cachefile} ./{name}")


def load_model(device: str = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model_parameters = {
        'num_outputs': 35, # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
        'pretrained': True,
        'backbone': 'efficientnet-b0',
        'pred_category': False,
        'curriculum_steps': [0, 0, 0, 0],
    }
    model_weights = 'model_tusimple_2695.pt'
    fetch_model_weights(model_weights)

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


def find_left_right_lanes(results, w: int, h: int):
    # results = outputs[0][0]

    lanePoints = []
    for _, raw_lane in enumerate(results):
        raw_lane = raw_lane.cpu().numpy()
        if raw_lane[0] == 0:  # Skip invalid lanes
            continue

        # generate points from the polynomial
        lane = raw_lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions
        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * h).astype(int)
        points[:, 0] = (np.polyval(lane, ys) * w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < w)]
        lanePoints.append(points)

    leftLane = None
    rightLane = None
    leftLaneX = -w
    leftLaneY = 0
    rightLaneX = w*2
    rightLaneY = 0

    for k in range(len(lanePoints)):
        points = lanePoints[k]

        xBottom = points[-1][0]
        yBottom = points[-1][1]

        # We're looking for the start point that is the most to the bottom (closer to the car in Y)
        # and closest in X too.
        if xBottom < w//2 and yBottom > leftLaneY and xBottom > leftLaneX:
            print("found new candidate left lane:", k)
            leftLaneX = xBottom
            leftLaneY = yBottom
            leftLane = points
        if xBottom > w//2 and yBottom > rightLaneY and xBottom < rightLaneX:
            print("found new candidate right lane:", k)
            rightLaneX = xBottom
            rightLaneY = yBottom
            rightLane = points

    # print a message but this script will crash
    if leftLane is None:
        print("we could not find the left lane!")
    if rightLane is None:
        print("we could not find the right lane!")

    # Center line: show where is the car going right now
    nPoints = len(points)
    centerX = np.ones(nPoints) * (w//2)
    centerY = np.linspace(0, h, nPoints)
    centerLane = np.array([[x,int(y)] for x,y in zip(centerX, centerY)], dtype=int)


    # Filter the part of the lane that is actually to the left/right or center
    # and hasn't crossed through the image yet.
    leftLane = np.array([pt for pt in leftLane if pt[0] < w//2])
    rightLane = np.array([pt for pt in rightLane if pt[0] > w//2])

    # filter points with common Y for each lane
    lowerRight = max(rightLane[:,1])
    upperRight = min(rightLane[:,1])
    lowerLeft = max(leftLane[:,1])
    upperLeft = min(leftLane[:,1])
    upper = max(upperRight, upperLeft)
    lower = min(lowerRight, lowerLeft)

    # bring down the upper limit more to discard effects of far horizon
    upper = upper + (lower - upper) // 3

    leftLane = np.array([pt for pt in leftLane if pt[1] <= lower and pt[1] >= upper])
    rightLane = np.array([pt for pt in rightLane if pt[1] <= lower and pt[1] >= upper])
    centerLane = np.array([pt for pt in centerLane if pt[1] <= lower and pt[1] >= upper])

    # measure distance
    leftDistance = similaritymeasures.area_between_two_curves(centerLane, leftLane)
    rightDistance = similaritymeasures.area_between_two_curves(centerLane, rightLane)
    area = leftDistance+rightDistance
    offset = (rightDistance-leftDistance) / (area)
    # print("Computed offset of the car relative to its lane:", offset)

    return offset, leftDistance, rightDistance, leftLane, rightLane, centerLane


def draw_overlays(img, results):
    h, w, _ = image.shape
    offset, leftDistance, rightDistance, leftLane, rightLane, centerLane = find_left_right_lanes(results, w, h)

    for current_point, next_point in zip(leftLane[:-1], leftLane[1:]):
        img = cv2.line(img, tuple(current_point), tuple(next_point), color=(255, 0, 0), thickness=2)
    for current_point, next_point in zip(rightLane[:-1], rightLane[1:]):
        img = cv2.line(img, tuple(current_point), tuple(next_point), color=(255, 0, 255), thickness=2)
    for current_point, next_point in zip(centerLane[:-1], centerLane[1:]):
        img = cv2.line(img, tuple(current_point), tuple(next_point), color=(0, 0, 128), thickness=2)

    cv2.putText(img,
            "L: %.0f" % leftDistance,
            (leftLane[-1][0], leftLane[-1][1] - 20),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.8,
            color=(128, 0, 0),
            thickness=2)

    cv2.putText(img,
                "R: %.0f" % rightDistance,
                (rightLane[-1][0] - 200, rightLane[-1][1] - 20),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(128, 0, 128),
                thickness=2)

    cv2.putText(img,
                "Offset %+d%%" % (offset*100),
                (w // 2 - 100, h // 2 - 20),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2)

    return img


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    sys.excepthook = log_on_exception

    image = load_image(sys.argv[1])
    outputs = run_inference(image)
    results = outputs[0][0]
    h, w, _ = image.shape
    offset, leftDistance, rightDistance, leftLane, rightLane, centerLane = find_left_right_lanes(results, w, h)

    print("Lane offset:", offset)
    draw_overlays(image, results)

    dstfile = sys.argv[1].replace(".jpg", "-lanes.jpg").replace(".png", "-lanes.png")
    cv2.imwrite(dstfile, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("Output written to", dstfile)
