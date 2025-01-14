import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import torch
import glob
import torch.utils.data as data
import sys
sys.path.append('.')
sys.path.append('..')
from utils import visualize_point_clouds
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LoadElectrode(data.Dataset):
    def __init__(self, landmark_dir, split='train'): 

        with open('./my_split/SSM/{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]
        
        lead_list = ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        self.metadata = list()
        unit = 0.01
        electrode_node_df = pd.read_csv(landmark_dir)
        case_ids = electrode_node_df['IDs'].tolist()
        for case_id in case_ids:
            if str(case_id) in filenames:
                electrode_node_data = electrode_node_df[electrode_node_df['IDs'] == int(case_id)]
                points_electrode_nodes = np.array([[electrode_node_data[f'{lead_name}x'].values[0],
                                electrode_node_data[f'{lead_name}y'].values[0],
                                electrode_node_data[f'{lead_name}z'].values[0]] for lead_name in lead_list])
                self.metadata.append((unit*points_electrode_nodes))

    def __getitem__(self, index):

        pc_electrode = self.metadata[index]
        pc_electrode_torch = torch.from_numpy(pc_electrode).float()

        return pc_electrode_torch

    def __len__(self):
        return len(self.metadata)
        
class LoadDataset(data.Dataset):
    def __init__(self, path, num_input, num_kp, num_resample = 1, split='train', load_SSM_gd = False): 
        self.path = path
        self.num_input = num_input
        self.num_kp = num_kp
        
        electrode_labels_list = ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        if split == 'train':
            num_resample = 30
        else:
            num_resample = 1
        
        with open('./my_split/{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]

        self.metadata = list()
        for filename in filenames:
            print(filename)
            datapath = path + '/' + filename + '/'

            unit = 0.1 # 0.01 # the coordinate is in mm
            cloud = pv.PolyData(datapath + 'torso_contour_full.vtk')
            nodesXYZ = unit*cloud.points
            label_index = cloud.point_data['Labels']
            mesh = pv.PolyData(datapath + 'torso_mesh.vtk')
            meshXYZ = unit*mesh.points

            pc_torso = nodesXYZ[label_index > 1]
            if load_SSM_gd:
                pc_electrode_gd = nodesXYZ[label_index == 1]  
            else:
                electrode_gd_path = os.path.join('./results/electrode_gd_200cases_by_yilin/after_' + filename + '.csv')
                if not os.path.exists(electrode_gd_path):
                    print('file missing: ' + filename)
                    continue    
                df = pd.read_csv(electrode_gd_path)
                df[['X', 'Y', 'Z']] *= unit
                label_to_coords = {label: coords for coords, label in zip(df[['X', 'Y', 'Z']].values, df['Label'])}
                pc_electrode_gd = np.array([label_to_coords[label] for label in electrode_labels_list])
                
            pc_torso_label = label_index[label_index > 1][..., np.newaxis]

            for i in range(num_resample):  # Sample 10 times
                # nodesXYZ_normalized = normalize_point_cloud(nodesXYZ)   

                # if  i > -1: #i%2 == 0:     # i > -1: 
                pc_torso_resampled, idx_remained = resample_pcd(pc_torso, self.num_input, seed=i)
                # pc_torso_label_resampled = pc_torso_label[idx_remained]
                # else:
                #     pc_torso_resampled, idx_remained = resample_pcd(meshXYZ, self.num_input*4)
                #     pc_torso_incomplete = random_gen_incomplete_point_cloud(pc_torso_resampled)
                #     pc_torso_resampled, idx_remained = resample_pcd(pc_torso_incomplete, self.num_input)
                
                pc_torso_label_resampled = np.zeros_like(pc_torso_label[idx_remained])
                
                pc_torso_labeled = np.concatenate((pc_torso_resampled, pc_torso_label_resampled), axis=1)
                mesh_torso_coarse, _ = resample_pcd(meshXYZ, self.num_input//2, seed=i)
                mesh_torso_dense, _ = resample_pcd(meshXYZ, 2*self.num_input, seed=i)

                # centroid_idx = farthest_point_sampling(meshXYZ, self.num_input//2, seed=i)
                # mesh_torso_coarse = gather_points(meshXYZ, centroid_idx)
                # centroid_idx = farthest_point_sampling(meshXYZ, 2*self.num_input, seed=i)
                # mesh_torso_dense = gather_points(meshXYZ, centroid_idx)

                centroid_idx = farthest_point_sampling(mesh_torso_coarse, self.num_input//8, seed=i)
                mesh_torso_keypoint = gather_points(mesh_torso_coarse, centroid_idx)

                self.metadata.append((pc_torso_labeled, pc_electrode_gd, mesh_torso_coarse, mesh_torso_dense, mesh_torso_keypoint))

    def __getitem__(self, index):
        pc_torso_labeled, pc_electrode, mesh_torso_coarse, mesh_torso_dense, mesh_torso_keypoint = self.metadata[index]

        pc_torso_labeled_torch = torch.from_numpy(pc_torso_labeled).float()
        pc_electrode_torch = torch.from_numpy(pc_electrode).float()

        mesh_torso_coarse_torch = torch.from_numpy(mesh_torso_coarse).float()
        mesh_torso_dense_torch = torch.from_numpy(mesh_torso_dense).float()
        mesh_torso_kp_torch = torch.from_numpy(mesh_torso_keypoint).float()

        return pc_torso_labeled_torch, pc_electrode_torch, mesh_torso_coarse_torch, mesh_torso_dense_torch, mesh_torso_kp_torch

    def __len__(self):
        return len(self.metadata)


def farthest_point_sampling(pcd, n_points, seed=0):
    """Farthest Point Sampling to select n_points from a point cloud."""
    np.random.seed(seed)
    N, _ = pcd.shape
    centroids = np.zeros(n_points, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(n_points):
        centroids[i] = farthest
        centroid = pcd[farthest, :].reshape(1, 3)
        dist = np.sum((pcd - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids

def gather_points(pcd, idx):
    """Gather points from a point cloud using the given indices."""
    return pcd[idx]

class LoadDataset_old(data.Dataset):
    def __init__(self, path, num_input, num_resample = 20, split='train'): 
        self.path = path
        self.num_input = num_input

        with open('./my_split/{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]

        self.metadata = list()
        for filename in filenames:
            print(filename)
            datapath = path + '/' + filename + '/'

            unit = 0.1 # 0.01 # the coordinate is in mm
            cloud = pv.PolyData(datapath + 'torso_contour_full.vtk')
            nodesXYZ = unit*cloud.points
            label_index = cloud.point_data['Labels']
            mesh = pv.PolyData(datapath + 'torso_mesh.vtk')
            meshXYZ = unit*mesh.points

            pc_torso = nodesXYZ[label_index > 1]
            pc_electrode = nodesXYZ[label_index == 1]        
            pc_torso_label = label_index[label_index > 1][..., np.newaxis]

            pc_torso = nodesXYZ[label_index > 1]
            pc_electrode = nodesXYZ[label_index == 1]        
            pc_torso_label = label_index[label_index > 1][..., np.newaxis]

            for i in range(num_resample):  # Sample 10 times
                # nodesXYZ_normalized = normalize_point_cloud(nodesXYZ)         
                pc_torso_resampled, idx_remained = resample_pcd(pc_torso, self.num_input, seed=i)
                pc_torso_label_resampled = pc_torso_label[idx_remained]
                pc_torso_labeled = np.concatenate((pc_torso_resampled, pc_torso_label_resampled), axis=1)
                mesh_torso_coarse, _ = resample_pcd(meshXYZ, self.num_input//2, seed=i)
                mesh_torso_dense, _ = resample_pcd(meshXYZ, 2*self.num_input, seed=i)
                mesh_torso_keypoint, _ = resample_pcd(meshXYZ, self.num_input//64, seed=i)

                self.metadata.append((pc_torso_labeled, pc_electrode, mesh_torso_coarse, mesh_torso_dense, mesh_torso_keypoint))

    def __getitem__(self, index):
        pc_torso_labeled, pc_electrode, mesh_torso_coarse, mesh_torso_dense, mesh_torso_keypoint = self.metadata[index]

        pc_torso_labeled_torch = torch.from_numpy(pc_torso_labeled).float()
        pc_electrode_torch = torch.from_numpy(pc_electrode).float()

        mesh_torso_coarse_torch = torch.from_numpy(mesh_torso_coarse).float()
        mesh_torso_dense_torch = torch.from_numpy(mesh_torso_dense).float()
        mesh_torso_keypoint_torch = torch.from_numpy(mesh_torso_keypoint).float()

        return pc_torso_labeled_torch, pc_electrode_torch, mesh_torso_coarse_torch, mesh_torso_dense_torch, mesh_torso_keypoint_torch

    def __len__(self):
        return len(self.metadata)

def resample_pcd(pcd, n, seed=0):
    """Drop or duplicate points so that pcd has exactly n points"""
    # idx = np.random.permutation(pcd.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])

    return pcd[idx[:n]], idx[:n]

# normalize point cloud based on apex coordinate
def normalize_point_cloud(point_cloud):
    # Step 1: Centering
    centroid = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - centroid

    # Step 2: Scaling
    max_extent = np.max(np.abs(centered_point_cloud), axis=0)
    scaled_point_cloud = centered_point_cloud / np.max(max_extent)

    return scaled_point_cloud

# Function to create a random axis-aligned bounding box
def random_bounding_box(min_bound, max_bound, max_size):
    center = np.random.uniform(min_bound, max_bound)
    size = np.random.uniform(0, max_size, size=3)
    min_corner = np.maximum(min_bound, center - size / 2)
    max_corner = np.minimum(max_bound, center + size / 2)
    return min_corner, max_corner

# Function to remove points within a bounding box
def remove_points_in_bounding_box(points, min_bound, max_bound):
    mask = np.all(np.logical_or(points < min_bound, points > max_bound), axis=1)
    removed_points = points[~mask]
    remaining_points = points[mask]
    return remaining_points, removed_points

def random_gen_incomplete_point_cloud(points):

    original_points = points.copy()

    # Get the bounding box of the entire point cloud
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    # Number of parts to remove
    num_parts_to_remove = random.randint(2, 4)  # Randomly choose between 1 and 5 

    # Maximum size of the region to remove
    max_size = (max_bound - min_bound) / 5   # Adjust based on your data

    all_removed_points = []
    # Remove random parts
    for _ in range(num_parts_to_remove):
        min_corner, max_corner = random_bounding_box(min_bound, max_bound, max_size)
        points, removed_points = remove_points_in_bounding_box(points, min_corner, max_corner)
        all_removed_points.append(removed_points)

    # points, idx_remained = resample_pcd(points, 1024) 
    # all_removed_points = np.vstack(all_removed_points) if all_removed_points else np.empty((0, 3))
    # visualize_point_clouds(original_points, points, all_removed_points)

    return points
    

if __name__ == '__main__':
    points = np.random.rand(1000, 3)
    
    unit = 0.1
    datapath = './dataset_mesh/1049499/'
    cloud = pv.PolyData(datapath + 'torso_mesh.vtk')
    pc_torso = unit*cloud.points
    pc_torso_resampled, idx_remained = resample_pcd(pc_torso, 1024*4)
    random_gen_incomplete_point_cloud(pc_torso_resampled)


    ROOT = './dataset/'
    GT_ROOT = os.path.join(ROOT, 'gt')
    PARTIAL_ROOT = os.path.join(ROOT, 'partial')

    train_dataset = LoadDataset(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='train')
    val_dataset = LoadDataset(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='val')
    test_dataset = LoadDataset(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='test')
    print("\033[33mTraining dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(train_dataset)))
    print("\033[33mValidation dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(val_dataset)))
    print("\033[33mTesting dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(test_dataset)))

    # visualization
    input_pc, coarse_pc, dense_pc, conditions = train_dataset[random.randint(0, len(train_dataset))-1]
    print("partial input point cloud has {} points".format(len(input_pc)))
    print("coarse output point cloud has {} points".format(len(coarse_pc)))
    print("dense output point cloud has {} points".format(len(dense_pc)))
