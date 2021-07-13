import os
import json
import argparse
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from config import *
import dataset
from model import LaneNet
from tqdm import tqdm
from utils.transforms import *
from utils.postprocess import *
from utils.prob2lines import getLane


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e",type=str)
    parser.add_argument("--Repredict", '-v', action="store_true", help="Regenerate predict")
    args = parser.parse_args()
    return args

# ------------ config ------------
args = parse_args()
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])
device = torch.device('cuda')


def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders

if args.Repredict:
    # ------------ data and model ------------
    # Imagenet mean, std
    mean=(0.3598, 0.3653, 0.3662)
    std=(0.2573, 0.2663, 0.2756)
    transform = Compose(Resize(resize_shape), ToTensor(),
                        Normalize(mean=mean, std=std))
    dataset_name = exp_cfg['dataset'].pop('dataset_name')
    Dataset_Type = getattr(dataset, dataset_name)
    test_dataset = Dataset_Type(Dataset_Path['CULane'], "test", transform)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=test_dataset.collate, num_workers=4)

    net = LaneNet(pretrained=True, **exp_cfg['model'])
    save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_best.pth')
    save_dict = torch.load(save_name, map_location='cpu')
    print("\nloading", save_name, "...... From Epoch: ", save_dict['epoch'])
    net.load_state_dict(save_dict['net'])
    net = torch.nn.DataParallel(net.to(device))
    net.eval()

    # ------------ test ------------
    out_path = os.path.join(exp_dir, "coord_output")
    evaluation_path = os.path.join(exp_dir, "evaluate")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    dump_to_json = []

    progressbar = tqdm(range(len(test_loader)))
    count_out_of_range = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            img = sample['img'].to(device)
            img_name = sample['img_name']

            output = net(img)
            embedding = output['embedding']
            binary_seg = output['binary_seg']
            embedding = embedding.detach().cpu().numpy()
            binary_seg = binary_seg.detach().cpu().numpy()
            for b in range(len(binary_seg)):
                embed_b = embedding[b]
                bin_seg_b = binary_seg[b]
                embed_b = np.transpose(embed_b, (1, 2, 0))
                bin_seg_b = np.argmax(bin_seg_b, axis=0)
                lane_seg_img = embedding_post_process(embed_b, bin_seg_b, 1.5)

                lane_coords = getLane.polyfit2coords_CULane(lane_seg_img, resize_shape=(590, 1640), y_px_gap=20, pts=18)
                for i in range(len(lane_coords)):
                    lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1], reverse=True)
                try:
                    lane_coords = sorted(lane_coords, key=lambda p: p[0])
                except Exception as e:
                    count_out_of_range += 1

                path_tree = split_path(img_name[b])
                save_dir, save_name = path_tree[-3:-1], path_tree[-1]
                save_dir = os.path.join(out_path, *save_dir)
                save_name = save_name[:-3] + "lines.txt"
                save_name = os.path.join(save_dir, save_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                with open(save_name, "w") as f:
                    for l in lane_coords:
                        for (x, y) in l:
                            print("{} {}".format(x, y), end=" ", file=f)
                        print(file=f)



            progressbar.update(1)
    progressbar.close()
    print("counter of out of range:{}".format(count_out_of_range))

# ---- evaluate ----
os.system("sh utils/lane_evaluation/CULane/Run.sh " + exp_name)