import argparse
import os
import time

from config import *
from utils.prob2lines import getLane
from model import LaneNet
from utils.transforms import *
from utils.postprocess import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str,help="Path to model weights")
    parser.add_argument("--band_width", '-b', type=float, default=1.5, help="Value of delta_v")
    parser.add_argument("--visualize", '-v', action="store_true", help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

    _set = "CULane"
    mean = IMG_MEAN[_set]
    std = IMG_STD[_set]
    transform_img = Resize((800, 288))
    transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
    transform = Compose(transform_img, transform_x)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB for net model input
    img = transform_img({'img': img})['img']
    x = transform_x({'img': img})['img']
    x.unsqueeze_(0)

    net = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
    save_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.eval()
    output = net(x)
    embedding = output['embedding']
    embedding = embedding.detach().cpu().numpy()
    embedding = np.transpose(embedding[0], (1, 2, 0))
    binary_seg = output['binary_seg']
    bin_seg_prob = binary_seg.detach().cpu().numpy()
    bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    seg_img = np.zeros_like(img)
    lane_seg_img = embedding_post_process(embedding, bin_seg_pred, args.band_width, 4)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        if lane_idx==0:
            continue
        seg_img[lane_seg_img == lane_idx] = color[i-1]
    img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    cv2.imwrite(os.path.join("visualization","{}_result.jpg".format(img_path.split("/")[-1].split(".")[0])), img)

    if args.visualize:
        cv2.imshow("", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    lane_coords = getLane.polyfit2coords_CULane(lane_seg_img, resize_shape=(590, 1640), y_px_gap=20, pts=18)
    for i in range(len(lane_coords)):
        lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1], reverse=True)


if __name__ == "__main__":
    main()
