import torch
import os
import numpy as np
from data.datasets import PointCloudDataset, my_collate_fn
import json
# test_dataset = PointCloudDataset(objects_dir)


def pad_collate_fn(batch):
    max_points = max([xyz.shape[0] for xyz, rgb in batch])
    padded_xyz = []
    padded_rgb = []
    for xyz, rgb in batch:
        pad_size = max_points - xyz.shape[0]
        padded_xyz.append(torch.cat([xyz, torch.zeros(pad_size, xyz.shape[1])], dim=0))
        padded_rgb.append(torch.cat([rgb, torch.zeros(pad_size, rgb.shape[1])], dim=0))
    
    return torch.stack(padded_xyz), torch.stack(padded_rgb)

objects_dir = "/root/V-DETR/results/objects"
# test_dataset = []
# for object_file in os.listdir(objects_dir):
#     xyz_rgb = np.load(os.path.join(objects_dir, object_file))
#     xyz = xyz_rgb[:, :3]
#     rgb = xyz_rgb[:, 3:6]
#     xyz = torch.from_numpy(xyz).float()
#     rgb = torch.from_numpy(rgb).float()
#     # print('pc.shape: ', xyz.shape)
#     print('rgb.shape: ', rgb)
#     test_dataset.append([xyz, rgb])

# test_dataset = PointCloudDataset(objects_dir)


# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=8, shuffle=False,
#     num_workers=1, pin_memory=True, sampler=None, drop_last=False, collate_fn=my_collate_fn
# )


with open(os.path.join("./Uni3D/data", 'templates.json')) as f:
    templates = json.load(f)["modelnet40_64"]

with open(os.path.join("./Uni3D/data", 'labels.json')) as f:
    labels = json.load(f)['scannet']