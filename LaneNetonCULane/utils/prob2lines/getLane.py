import cv2
import numpy as np


def getLane_CULane(prob_map, y_px_gap, pts, thresh, resize_shape=None):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)
    Return:
    ----------
    coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
    """
    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape

    coords = np.zeros(pts)
    for i in range(pts):
        y = int(h - i * y_px_gap / H * h - 1) # culane: int(h - i * y_px_gap / H * h - 1)
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(pts)
    return coords

def polyfit2coords_CULane(lane_pred, crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=2):
    if resize_shape is None:
        resize_shape = lane_pred.shape
        crop_h = 0
    h, w = lane_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)

    for i in [idx for idx in np.unique(lane_pred) if idx!=0]:
        ys_pred, xs_pred = np.where(lane_pred==i)

        poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
        ys = np.array([h-y_px_gap/(H-crop_h)*h*i for i in range(1, pts+1)])
        xs = np.polyval(poly_params, ys)

        y_min, y_max = np.min(ys_pred), np.max(ys_pred)
        if len([[int(x/w*W),H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts)) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max]) < 2:
            continue
        coordinates.append([[int(x/w*W),
                             H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts)) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max])

    return coordinates