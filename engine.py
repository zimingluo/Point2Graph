# Copyright (c) V-DETR authors. All Rights Reserved.
import torch
import datetime
import logging
import math
import time
import sys
import os
import json
import numpy as np
from torch.distributed.distributed_c10d import reduce
from util.ap_calculator import APCalculator
from util.misc import SmoothedValue
from util.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
    batch_dict_to_cuda,
)
from util.box_util import (flip_axis_to_camera_tensor, get_3d_box_batch_tensor)
from util.o3d_helper import visualize_pcd, visualize_3d_detection
import open3d as o3d
import sys
sys.path.append('./Uni3D/')
from Uni3D.main import inference
import open_clip
import Uni3D.model.uni3d as models



def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        if args.lr_scheduler == 'cosine':
            # Cosine Learning Rate Schedule
            curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
                1 + math.cos(math.pi * curr_epoch_normalized)
            )
        else:
            step_1, step_2 = args.step_epoch.split('_')
            step_1, step_2 = int(step_1), int(step_2)
            if curr_epoch_normalized < (step_1 / args.max_epoch):
                curr_lr = args.base_lr
            elif curr_epoch_normalized < (step_2 / args.max_epoch):
                curr_lr = args.base_lr / 10
            else:
                curr_lr = args.base_lr / 100
    return curr_lr

def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
):
    ap_calculator = None

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device
    
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        batch_data_label = batch_dict_to_cuda(batch_data_label,local_rank=net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        if args.use_superpoint:
            inputs["superpoint_per_point"] = batch_data_label["superpoint_labels"]
        outputs = model(inputs)
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            eta_seconds = (max_iters - curr_iter) * (time.time() - curr_time)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; ETA {eta_str}"
            )
        
        curr_iter += 1
        barrier()

    return ap_calculator, curr_iter, curr_lr, loss_avg.avg, loss_dict_reduced


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        no_nms=args.test_no_nms,
        args=args
    )

    curr_iter = 0
    device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""



    # It is recommended to download clip model in advance and then load from the local
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name="EVA02-E-14-plus", pretrained="./Uni3D/downloads/open_clip_pytorch_model.bin") 
    clip_model.to(device)

    # create model
    uni3d_model = getattr(models, 'create_uni3d')(args=args)
    uni3d_model.to(device)


    for batch_idx, batch_data_label in enumerate(dataset_loader):   
        batch_data_label = batch_dict_to_cuda(batch_data_label,local_rank=device)
            
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        scene_name = batch_data_label['scan_name'][0]
        outputs = model(inputs) 

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)
            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"
        else:
            loss_dict_reduced = None

        if args.cls_loss.split('_')[0] == "focalloss":
            outputs["outputs"]["sem_cls_prob"] = outputs["outputs"]["sem_cls_prob"].sigmoid()

        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        if args.axis_align_test:
            outputs["outputs"]["box_corners"] = outputs["outputs"]["box_corners_axis_align"]

        batch_gt_map_cls, batch_pred_map_cls = ap_calculator.step_meter(outputs, batch_data_label)
        batch_pred_map_cls = batch_dict_to_cuda(batch_pred_map_cls,local_rank=device)

        if is_primary() and curr_iter % args.log_every == 0:
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]"
            )
        curr_iter += 1
        barrier()

        ## compute bounding box
        gt_boxs = [gt_box[:,[0,2,1]] for gt_cls, gt_box in batch_gt_map_cls[0]]
        pred_boxs = [pred_box[:,[0,2,1]] for pred_cls, pred_box, con in batch_pred_map_cls[0] if con > args.conf_thresh]


        for pred_box in pred_boxs:
            pred_box[:, 2] = - pred_box[:, 2]
        detected_objects = []
        if args.inference_only:


            ply_file = os.path.join(args.dataset_root_dir, 'scans', scene_name, scene_name+'_vh_clean_2.ply')
            mesh = o3d.io.read_triangle_mesh(ply_file)

            meta_file = os.path.join(args.dataset_root_dir, 'scans', scene_name, scene_name+'.txt')
            lines = open(meta_file).readlines()
            for line in lines:
                if 'axisAlignment' in line:
                    axis_align_matrix = [float(x) \
                        for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                    break
            axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
            mesh.transform(axis_align_matrix)


            for i, pred_box in enumerate(pred_boxs):
                
                bbox_min = np.min(pred_box, axis=0) 
                bbox_max = np.max(pred_box, axis=0)
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
                cropped_mesh = mesh.crop(bounding_box)
                try:
                    cropped_object = cropped_mesh.sample_points_uniformly(number_of_points=args.npoints)
                except RuntimeError as e:
                    print("Input mesh has no triangles.")
                    continue
                cropped_points = np.asarray(cropped_object.points)
                cropped_colors = np.asarray(cropped_object.colors)
                cropped_object = np.hstack((cropped_points, cropped_colors))

                cropped_pcd = o3d.geometry.PointCloud()
                cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
                cropped_pcd.colors = o3d.utility.Vector3dVector(cropped_colors)
                labels = np.array(cropped_pcd.cluster_dbscan(eps=args.eps, min_points=args.min_points, print_progress=False))

                num_clusters = labels.max() + 1

                # Find the largest cluster
                largest_cluster_label = max(range(num_clusters), key=lambda x: np.sum(labels == x))
                largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

                # Extract points belonging to the largest cluster
                filtered_points = cropped_object[largest_cluster_indices]
                if len(filtered_points) < args.npoints / 2:
                    detected_objects.append(cropped_object)
                    np.save(f"./results/objects/object_{i}.npy", cropped_object)
                else:
                    detected_objects.append(filtered_points)
                    np.save(f"./results/objects/object_{i}.npy", cropped_object)

    return ap_calculator, detected_objects


