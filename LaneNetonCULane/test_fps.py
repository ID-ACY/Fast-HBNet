import argparse
import json
import os
import shutil
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
import dataset
from model import LaneNet
from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from utils.postprocess import embedding_post_process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e",type=str)

    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]
num_val = 9765
with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

device = torch.device(exp_cfg['device'])
tensorboard = TensorBoard(exp_dir)

# ------------ train data ------------
# # CULane mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)

dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)

# ------------ val data ------------
transform_val = Compose(Resize(resize_shape), ToTensor(),
                        Normalize(mean=mean, std=std))
val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=val_dataset.collate, num_workers=4)

# ------------ preparation ------------
net = LaneNet(pretrained=True, **exp_cfg['model'])
net = net.to(device)
net = torch.nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
lr_scheduler = PolyLR(optimizer, 0.9, exp_cfg['MAX_ITER'])


def val():
    net.eval()
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)

            output = net(img, segLabel)

            loss = output['loss']
            if isinstance(net, torch.nn.DataParallel):
                loss = output['loss'].sum()

            progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)

    progressbar.close()


    print("------------------------\n")


def main():
    global best_val_loss
    save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
    if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
    else:
            net.load_state_dict(save_dict['net'])
    optimizer.load_state_dict(save_dict['optim'])
    lr_scheduler.load_state_dict(save_dict['lr_scheduler'])

    print("\nValidation For Experiment: ", exp_dir)
    print(time.strftime('%H:%M:%S', time.localtime()))
    start_time = time.time()
    val()
    print("the fps of LaneNet is:{}".format(num_val/(time.time()-start_time)))


if __name__ == "__main__":
    main()
