import numpy as np
import open3d as o3d
import pyviz3d.visualizer as viz
from typing import Optional, Tuple, Union
import os 
from scipy.spatial.transform import Rotation as R

COLORS = [[.8, .1, .1], [.1, .1, .8], [.1, .8, .1]]     # TODO: modify this


def extract_scale_translation_rotation(axis_align_matrix):
    # Extract translation
    translation = axis_align_matrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = axis_align_matrix[:3, :3]

    # Compute scale factors
    scale_factors = np.linalg.norm(rotation_matrix, axis=0)

    # Normalize the rotation matrix to remove scale
    normalized_rotation_matrix = rotation_matrix / scale_factors

    # Convert rotation matrix to Euler angles and then to quaternion
    rotation = R.from_matrix(normalized_rotation_matrix)
    quaternion = rotation.as_quat()

    return scale_factors, translation, quaternion


def visualize_3d_detection(scene_name, point_cloud, bounding_boxes, labels=None, vis_type='mesh'):
    # Create a PyViz3D visualizer
    v = viz.Visualizer()

    # Add the mesh file
    if vis_type=='mesh':
        meta_file = os.path.join('./scannet/scans/', 'scans', scene_name, scene_name+'.txt')
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

        scale_factors, translation, quaternion = extract_scale_translation_rotation(axis_align_matrix)

        v.add_mesh(scene_name,
                path=os.path.join('./scannet/scans/', scene_name, scene_name+'_vh_clean_2.ply'),
                rotation=viz.euler_to_quaternion(quaternion),
                scale=np.array(scale_factors),
                translation=np.array(translation),
                # color=np.array([50, 225, 50])
            )

        v.add_mesh(scene_name, path=os.path.join('./scannet/scans/', scene_name, scene_name+'_vh_clean_2.ply'))

    elif vis_type=='point':

        # Add the point cloud
        point_positions = point_cloud[:, :3]
        point_colors = (point_cloud[:, 3:6] * 255).astype(np.uint8)  # Assuming RGB values are in [0, 1]

        
        point_positions = np.asarray(point_cloud.points)
        point_colors = np.asarray(point_cloud.colors)
        point_size = 30.0  # Adjust the point size as needed
        v.add_points(scene_name, point_positions, point_colors, point_size=point_size)
    
    type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
    class2type = {v: k for k, v in type2class.items()}

    # Define a color map for different labels
    color_map = {
        0: np.array([255, 0, 0]),       # Red
        1: np.array([0, 255, 0]),       # Green
        2: np.array([0, 0, 255]),       # Blue
        3: np.array([255, 255, 0]),     # Yellow
        4: np.array([255, 0, 255]),     # Magenta
        5: np.array([0, 255, 255]),     # Cyan
        6: np.array([128, 0, 0]),       # Maroon
        7: np.array([0, 128, 0]),       # Dark Green
        8: np.array([0, 0, 128]),       # Navy
        9: np.array([128, 128, 0]),     # Olive
        10: np.array([128, 0, 128]),    # Purple
        11: np.array([0, 128, 128]),    # Teal
        12: np.array([192, 192, 192]),  # Silver
        13: np.array([128, 128, 128]),  # Gray
        14: np.array([255, 165, 0]),    # Orange
        15: np.array([255, 20, 147]),   # Deep Pink
        16: np.array([0, 191, 255]),    # Deep Sky Blue
        17: np.array([34, 139, 34]),    # Forest Green
    }

    # Add the bounding boxes
    for i, (bbox, label) in enumerate(zip(bounding_boxes, labels)):
        # Create an Axis-Aligned Bounding Box from the given vertices

        bbox_min = np.min(bbox, axis=0)
        bbox_max = np.max(bbox, axis=0)

        color = color_map.get(label, [0, 255, 0])  # Default color is white if label not found
        name = class2type.get(label, 'None')
        # position = bbox[0]
        # size = bbox[1] - bbox[0]
        position = (bbox_max + bbox_min) / 2.0
        size = bbox_max - bbox_min
        v.add_bounding_box(name, position=position, size=size, color=color)

    # Save the visualization
    v.save('./visualization/' + scene_name)


def visualize_mesh(mesh, pred_boxs, predicted_labels, scane_name):

    def vertices_in_bbox(vertices, bbox):
        """
        Determine which vertices of the mesh fall inside the bounding box.

        Parameters:
        - vertices: An array of mesh vertices.
        - bbox: A bounding box defined by eight vertices.

        Returns:
        - mask: A boolean mask array indicating which vertices are inside the bounding box.
        """
        # Extract the min and max coordinates from the bounding box vertices
        xmin, ymin, zmin = np.min(bbox, axis=0)
        xmax, ymax, zmax = np.max(bbox, axis=0)
        
        mask = (vertices[:, 0] >= xmin) & (vertices[:, 0] <= xmax) & \
               (vertices[:, 1] >= ymin) & (vertices[:, 1] <= ymax) & \
               (vertices[:, 2] >= zmin) & (vertices[:, 2] <= zmax)
        return mask

    vertices = np.asarray(mesh.vertices)
    colors = np.ones_like(vertices) * 0.5


    # Generate a unique color for each label using a colormap
    unique_labels = list(set(predicted_labels))
    colormap = cm.get_cmap('tab20', len(unique_labels))
    # label_to_color = {label: colormap(i)[:3] for i, label in enumerate(unique_labels)}
    label_to_color = {
        0: np.array([255, 0, 0]),       # Red
        1: np.array([0, 255, 0]),       # Green
        2: np.array([0, 0, 255]),       # Blue
        3: np.array([255, 255, 0]),     # Yellow
        4: np.array([255, 0, 255]),     # Magenta
        5: np.array([0, 255, 255]),     # Cyan
        6: np.array([128, 0, 0]),       # Maroon
        7: np.array([0, 128, 0]),       # Dark Green
        8: np.array([0, 0, 128]),       # Navy
        9: np.array([128, 128, 0]),     # Olive
        10: np.array([128, 0, 128]),    # Purple
        11: np.array([0, 128, 128]),    # Teal
        12: np.array([192, 192, 192]),  # Silver
        13: np.array([128, 128, 128]),  # Gray
        14: np.array([255, 165, 0]),    # Orange
        15: np.array([255, 20, 147]),   # Deep Pink
        16: np.array([0, 191, 255]),    # Deep Sky Blue
        17: np.array([34, 139, 34]),    # Forest Green
    }


    # Iterate over each predicted bounding box and label
    for bbox, label in zip(pred_boxs, predicted_labels):
        mask = vertices_in_bbox(vertices, bbox)
        color = label_to_color[label] / 255 # Get the color corresponding to the label
        colors[mask] = color

    # Assign colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Visualize the result
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(f"./visualization/{scane_name}_ObjectDetection.ply", mesh)


def visualize_pcd(
    point_cloud: Optional[Union[np.array, Tuple]]=None, 
    bboxes: Optional[Tuple[np.array]]=None
):
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5],
                      [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]])
    visualization_group = []
    
    if point_cloud is not None:
        if not isinstance(point_cloud, tuple):
            point_cloud = (point_cloud, )
        for pc in point_cloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            if pc.shape[-1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])
            else:
                pcd.colors = o3d.utility.Vector3dVector(
                    np.ones_like(pc[:, :3]) * [0.8, 0.8, 0.8]
                )
            if pc.shape[-1] >= 9:
                pcd.normals = o3d.utility.Vector3dVector(pc[:, 6:9])
            visualization_group.append(pcd)

    if bboxes is not None:
        if not isinstance(bboxes, tuple):
            bboxes = (bboxes,)
        for idx, boxgroup in enumerate(map(np.array, bboxes)):
            corners = boxgroup.reshape(-1, 3)
            edges = lines[None, ...] \
                    + (np.ones_like(lines[None]).repeat(boxgroup.shape[0], axis=0)
                       * np.arange(0, len(corners), boxgroup.shape[1])[:, None, None])
            edges = edges.reshape(-1, 2)
            # bounding box corners and bounding box edges
            box_corner = o3d.geometry.PointCloud()
            box_corner.points = o3d.utility.Vector3dVector(corners)
            box_corner.colors = o3d.utility.Vector3dVector(
                np.ones_like(corners) * COLORS[idx]
            )
            box_edge = o3d.geometry.LineSet()
            box_edge.lines = o3d.utility.Vector2iVector(edges)
            box_edge.colors = o3d.utility.Vector3dVector(
                np.ones((len(edges), 3)) * COLORS[idx]
            )
            box_edge.points = o3d.utility.Vector3dVector(corners)
            # store #
            visualization_group.extend([box_corner, box_edge])

    o3d.visualization.draw_plotly(visualization_group)
    return None
