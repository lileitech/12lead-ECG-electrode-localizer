import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import glob
from torch.autograd import Function
# from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
EPS = 1e-4

def plot_pcd(ax, pcd):
    """ Plot input point cloud """
    # ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], s=10, c='#FDDA99', label='prediction') 
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=10, c='#A2DCEC', label='ground truth') 
    ax.set_axis_off()

def chamfer_distance(point_cloud1, point_cloud2):
    # Calculate distances from each point in cloud1 to every point in cloud2
    distances1_to_2 = np.linalg.norm(point_cloud1[:, np.newaxis, :] - point_cloud2, axis=2)

    # Calculate distances from each point in cloud2 to every point in cloud1
    distances2_to_1 = np.linalg.norm(point_cloud2[:, np.newaxis, :] - point_cloud1, axis=2)

    # Calculate Chamfer distance
    chamfer_dist = np.mean(np.mean(np.min(distances1_to_2, axis=1)) + np.mean(np.min(distances2_to_1, axis=1)))

    return chamfer_dist

def init_graph(shape_xyz, skel_xyz):
    '''
    :param shape_xyz: size[B, n_coarse, 3]
    :param skel_xyz: size[B, n_keypoint, 3]

    :return: adjacency_matrix: size[B, n_keypoint, n_keypoint]
    '''

    n_batch, n_keypoints = skel_xyz.size()[0], skel_xyz.size()[1]

    knn_skel = knn_with_batch(skel_xyz, skel_xyz, n_keypoints, is_max=False)
    knn_sp2sk = knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)

    A = torch.zeros((n_batch, n_keypoints, n_keypoints)).float().to(skel_xyz.device)

    # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
    A[torch.arange(n_batch)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
    A[torch.arange(n_batch)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1

    # initialize A with topology prior
    A[torch.arange(n_batch)[:, None, None], torch.arange(n_keypoints)[None, :, None], knn_skel[:, :, 1:3]] = 1
    A[torch.arange(n_batch)[:, None, None], knn_skel[:, :, 1:3], torch.arange(n_keypoints)[None, :, None]] = 1

    adjacency_matrix = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1,2), diagonal=1).transpose(1,2)

    return adjacency_matrix

def gather_features(features, indices):
    """
    Gathers the features corresponding to the given indices.

    Input:
        features: feature data, [B, N, n_features]
        indices: sampled point indices, [B, npoint]
    Return:
        sampled_features: sampled feature data, [B, npoint, n_features]
    """
    B, N, n_features = features.shape
    npoint = indices.shape[1]
    indices = torch.clamp(indices, 0, N - 1)
    batch_indices = torch.arange(B, dtype=torch.long).view(B, 1).to(features.device)
    batch_indices = batch_indices.repeat(1, npoint)
    sampled_features = features[batch_indices, indices, :]
    return sampled_features

def farthest_feature_sample(features, npoint):
    """
    Input:
        features: feature data, [B, N, n_features] -- B, 1024, 2048
        npoint: number of samples --B, 64
    Return:
        centroids: sampled point indices, [B, npoint]
    """
    device = features.device
    B, N, n_features = features.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = features[batch_indices, farthest, :].view(B, 1, n_features)
        dist = torch.sum((features - centroid) ** 2, -1) # here, this is not Euclidean distance anymore as the dimention of features is not 3， what is it then？ 
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    # Ensure indices are within bounds
    centroids = torch.clamp(centroids, 0, N - 1)
    
    return centroids

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# concat triangle and skeleton, graph as input
def Batch_Keypoint_graph_interpolation_v2(batch_keypoint, graph, max_n1=1024, max_n2=512):  # batch_keypoint: (B, k, 3)
    n_batch = batch_keypoint.size()[0]
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(n_batch).unsqueeze(-1).unsqueeze(-1), vertices_index]
    return random_sample_triangle_skeleton(triangles, batch_keypoint, max_n1=max_n1, max_n2=max_n2)

def Batch_calculate_area(closest_keypoints):  # (B, n, 3, 3)
    closest_keypoints = closest_keypoints.double()

    side_ab = torch.sqrt((closest_keypoints[..., 0, 0] - closest_keypoints[..., 1, 0]) ** 2 + \
                         (closest_keypoints[..., 0, 1] - closest_keypoints[..., 1, 1]) ** 2 + \
                         (closest_keypoints[..., 0, 2] - closest_keypoints[..., 1, 2]) ** 2)

    side_ac = torch.sqrt((closest_keypoints[..., 0, 0] - closest_keypoints[..., 2, 0]) ** 2 + \
                         (closest_keypoints[..., 0, 1] - closest_keypoints[..., 2, 1]) ** 2 + \
                         (closest_keypoints[..., 0, 2] - closest_keypoints[..., 2, 2]) ** 2)

    side_bc = torch.sqrt((closest_keypoints[..., 1, 0] - closest_keypoints[..., 2, 0]) ** 2 + \
                         (closest_keypoints[..., 1, 1] - closest_keypoints[..., 2, 1]) ** 2 + \
                         (closest_keypoints[..., 1, 2] - closest_keypoints[..., 2, 2]) ** 2)

    s = (side_ab + side_bc + side_ac) / 2
    # print(s.dtype)
    # area = torch.sqrt(s.float() * (s - side_ab).float()).double() * \
    #        torch.sqrt((s - side_bc).float() * (s - side_ac).float()).double()

    area = (s * (s - side_ab) * (s - side_bc) * (s - side_ac) + 1e-14) ** 0.5
    # print(area.dtype)

    return area.float(), side_ab.float(), side_bc.float(), side_ac.float()  # (B, n)

def random_sample_triangle_skeleton(triangle, keypoint, max_n1, max_n2, interval=0.001):  # (B, n, 3, 3)
    ver_a = triangle[..., 0, :].unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ab = (triangle[..., 1, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ac = (triangle[..., 2, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)

    area, side_ab, side_bc, side_ac = Batch_calculate_area(triangle)
    c_square = (side_ab + side_bc + side_ac) ** 2
    c_square = (c_square/ interval).long()
    count = torch.clamp(c_square, 0, 100)  # (B, n)
    count = count.unsqueeze(-1)  # (B, n, 1)

    base_index = torch.arange(100).expand(count.size()[0], count.size()[1], 100).to(triangle.device)
    binary_mask = base_index < count  # (B, n, 100)

    random_x = torch.rand(count.size()[0], count.size()[1], 100).to(triangle.device)
    random_y = torch.rand(count.size()[0], count.size()[1], 100).to(triangle.device)
    x = torch.where(random_x + random_y > 1.0, 1.0 - random_x, random_x)
    y = torch.where(random_x + random_y > 1.0, 1.0 - random_y, random_y)
    final_x = torch.where(binary_mask, x, torch.Tensor([0.0]).to(triangle.device)).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_y = torch.where(binary_mask, y, torch.Tensor([0.0]).to(triangle.device)).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_a = torch.where(binary_mask, torch.Tensor([1.0]).to(triangle.device), torch.Tensor([0.0]).to(triangle.device)).unsqueeze(-1).expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3)  # (B, n, 100, 3)
    coarse = ver_a * final_a + final_x * edge_ab + final_y * edge_ac
    coarse = coarse.reshape(coarse.size()[0], -1, 3)
    # print("coarse shape:", coarse.shape)
    # print("coarse:", coarse)
    final_coarse1 = torch.Tensor([]).to(triangle.device)
    final_coarse2 = torch.Tensor([]).to(triangle.device)
    for i in range(coarse.size()[0]):
        non_empty_mask = coarse[i].abs().sum(dim=-1).bool()  # filter out
        indices = non_empty_mask.nonzero(as_tuple=False).squeeze(-1)
        coarse_i = torch.cat((coarse[i][indices], keypoint[i]), 0)
        coarse_i = coarse_i.unsqueeze(0)
        # print("coarse_i:", coarse_i.shape)
        # resample
        idx = np.random.permutation(coarse_i.shape[1])
        if idx.shape[0] < max_n1:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n1 - coarse_i.shape[1])])
        coarse_i_1 = coarse_i[:, idx[:max_n1]]
        if idx.shape[0] < max_n2:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n2 - coarse_i.shape[1])])
        coarse_i_2 = coarse_i[:, idx[:max_n2]]
        final_coarse1 = torch.cat((final_coarse1, coarse_i_1), 0)
        final_coarse2 = torch.cat((final_coarse2, coarse_i_2), 0)
    return final_coarse1, final_coarse2
    
def batch_triangle_line_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    # print("A: ", A)
    A_2 = torch.bmm(A, A)
    A_3 = torch.bmm(A_2, A)
    tri_count = torch.diagonal(A_3, dim1=-2, dim2=-1).sum(1) / 6  # (B), triangle num of each sample
    # print("tri_count:", tri_count)
    triu_A = torch.triu(A)
    line_count = triu_A.sum((2, 1))
    # print("line_count:", line_count)
    count = (tri_count + line_count).long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num

def generate_triangle_vertices_v2(keypoint, A):
    # print(A)
    n_batch = A.size()[0]
    kp_num = A.size()[1]
    triu_A = torch.triu(A)
    count, max_num = batch_triangle_line_num(A)
    vertices = torch.LongTensor([]).to(keypoint.device)
    # print(count, max_num)
    '''generate triangle vertices'''
    edge_indices = triu_A.nonzero(as_tuple=False)
    u = edge_indices[:, [0, 1]]  # first index
    v = edge_indices[:, [0, 2]]  # second index
    e = u * torch.Tensor([[1, kp_num]]).to(keypoint.device) + v * torch.Tensor([[0, 1]]).to(keypoint.device)  # edge index (u * kp_num + v)
    e = e[:, 1].long()
    # u_e = torch.cat((u, e[:, 1].unsqueeze(1)), -1)
    # u_v = torch.cat((v, e[:, 1].unsqueeze(1)), -1)
    batch_index = u[:, 0].long()
    u_index = u[:, 1].long()
    v_index = v[:, 1].long()
    # print("batch_index: ", batch_index)
    edge_matrix = torch.zeros(n_batch, kp_num, kp_num * kp_num).to(keypoint.device)
    edge_matrix = edge_matrix.index_put([batch_index, u_index, e], torch.tensor(1.).to(keypoint.device))
    edge_matrix = edge_matrix.index_put([batch_index, v_index, e], torch.tensor(1.).to(keypoint.device))
    # print("edge_matrix: ", edge_matrix)
    final_matrix = torch.bmm(A, edge_matrix)
    tri_position = (final_matrix == 2).nonzero(as_tuple=False)  # return indices where items equal to 2.
    # print("tri_position: ", tri_position)
    u_origin = torch.div(tri_position[:, 2].float(), kp_num).floor().long()
    v_origin = torch.remainder(tri_position[:, 2], kp_num)
    triangle_vertices = torch.cat((tri_position[:, :2], u_origin.unsqueeze(1), v_origin.unsqueeze(1)), 1)

    '''generate skeleton vertices, and expand (a,b) to (a,b,b) to fit the format of triangle vertices'''
    line_vertices = torch.cat((edge_indices, edge_indices[:, 2].unsqueeze(1)), -1)
    line_batch_info = line_vertices[:, 0]  # items represent batch num
    triangle_batch_info = triangle_vertices[:, 0]  # items represent batch num
    # print("info:", triangle_batch_info)

    for i in range(n_batch):
        triangle_batch_i = (triangle_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        # print(triangle_batch_i)
        triangle_vertices_i = triangle_vertices[triangle_batch_i]  # triangle vertices of batch i
        # print("triangle_vertices: ", triangle_vertices_i.shape)
        line_batch_i = (line_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        line_vertices_i = line_vertices[line_batch_i]  # line vertices of batch i

        all_vertices_i = torch.cat((triangle_vertices_i[:, 1:], line_vertices_i[:, 1:]), 0)
        all_vertices_i, _ = torch.sort(all_vertices_i, dim=-1)
        all_vertices_i = all_vertices_i.unique(dim=0)
        # print("all_vertices_i:", all_vertices_i.shape)
        all_vertices_i = torch.cat(
            (all_vertices_i, torch.LongTensor([[0, 0, 0]]).to(keypoint.device).expand(max_num - count[i].item(), 3)), 0)
        # print("all_vertices_i:", all_vertices_i.shape)
        vertices = torch.cat((vertices, all_vertices_i.unsqueeze(0)), 0)
    return vertices.long()

def knn_with_batch(p1, p2, k, is_max=False):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :param is_max: k-nearest neighbors or k-farthest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    top_dist, k_nn = torch.topk(dist, k, dim=2, largest=is_max)

    return k_nn

def plot_two_pcd_with_edge(ax, pcd1, pcd2):
    """ Plot input point cloud with indices and connecting lines """

    # Plot and connect points for LV_endo
    ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], s=10, c='#FDDA99', label='prediction')
    for i in range(len(pcd1) - 1):
        ax.plot([pcd1[i, 0], pcd1[i + 1, 0]], [pcd1[i, 1], pcd1[i + 1, 1]], [pcd1[i, 2], pcd1[i + 1, 2]], color='#FDDA99')

    # Plot and connect points for LV_epi
    ax.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], s=10, c='#A2DCEC', label='gd')
    for i in range(len(pcd2) - 1):
        ax.plot([pcd2[i, 0], pcd2[i + 1, 0]], [pcd2[i, 1], pcd2[i + 1, 1]], [pcd2[i, 2], pcd2[i + 1, 2]], color='#A2DCEC')

    ax.set_axis_off()
    # ax.legend()

def plot_two_pcd_with_index(ax, pcd1, pcd2):
    """ Plot input point cloud with indices """
    
    ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], s=20, c='#FDDA99', label='prediction') 
    for i, (x, y, z) in enumerate(pcd1):
        ax.text(x, y, z, str(i), fontsize=10, color='#FDDA99', ha='center', va='center')

    ax.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], s=20, c='#A2DCEC', label='ground truth') 
    for i, (x, y, z) in enumerate(pcd2):
        ax.text(x, y, z, str(i), fontsize=10, color='#A2DCEC', ha='center', va='center')

    ax.set_axis_off()
    ax.legend()

def plot_two_pcd(ax, pcd1, pcd2):
    """ Plot input point cloud """
    ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], s=10, c='#FDDA99', label='prediction') 
    ax.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], s=10, c='#A2DCEC', label='ground truth') 
    ax.set_axis_off()

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar):
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)

        return pd_mu, pd_logvar

class alphaProductOfExperts(nn.Module):
    """Return parameters for weighted product of independent experts (mmJSD implementation).
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weights=None):
        if weights is None:
            num_components = mu.shape[0]
            weights = (1/num_components) * torch.ones(mu.shape).to(mu.device)
    
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        weights = torch.broadcast_to(weights, mu.shape)
        pd_var = 1. / torch.sum(weights * T + EPS, dim=0)
        pd_mu = pd_var * torch.sum(weights * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        
        return pd_mu, pd_logvar
    
class weightedProductOfExperts(nn.Module):
    """Return parameters for weighted product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weight):

        var = torch.exp(logvar) + EPS     
        weight = weight[:, None, :].repeat(1, mu.shape[1],1)
        T = 1.0 / (var + EPS)
        pd_var = 1. / torch.sum(weight * T + EPS, dim=0)
        pd_mu = pd_var * torch.sum(weight * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        return pd_mu, pd_logvar

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    
    Args:
    mu (torch.Tensor): Mean of distributions. M x D for M views.
    logvar (torch.Tensor): Log of Variance of distributions. M x D for M views.
    """

    def forward(self, mu, logvar):
        mean_mu = torch.mean(mu, axis=0)
        mean_logvar = torch.mean(logvar, axis=0)
        
        return mean_mu, mean_logvar

def visualize_PC_with_twolabel_rotated(nodes_xyz_pre, labels_pre, labels_gd, filename='PC_label.pdf'):
    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}

    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_gd = [color_dict[label] for label in labels_gd]
    colors_pre = [color_dict[label] for label in labels_pre]
    

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=1.5)  
    ax1.set_title('Ground truth')
    ax2.scatter(df['x'], df['y'], df['z'], c=colors_pre, s=1.5) 
    ax2.set_title('Prediction')
    ax1.set_axis_off() # Hide coordinate space 
    ax2.set_axis_off() # Hide coordinate space

    # 定义交互事件函数
    def on_rotate(event):
        # 获取当前旋转的角度
        elev = ax1.elev
        azim = ax1.azim
        
        # 设置两个子图的视角
        ax1.view_init(elev=elev, azim=azim)
        ax2.view_init(elev=elev, azim=azim)
        
        # 更新图形
        fig.canvas.draw()

    # 绑定交互事件
    fig.canvas.mpl_connect('motion_notify_event', on_rotate)

    plt.show()

def visualize_PC_with_twolabel(nodes_xyz_pre, labels_pre, labels_gd, filename='PC_label.pdf'):
    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}

    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_pre = [color_dict[label] for label in labels_pre]
    colors_gd = [color_dict[label] for label in labels_gd]

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_pre, s=1.5)  
    ax1.set_axis_off() # Hide coordinate space
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=1.5)    
    ax2.set_axis_off() # Hide coordinate space
    plt.subplots_adjust(wspace=0)
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)

# Visualize point clouds side by side
def visualize_point_clouds(original_points, remaining_points, removed_points):
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], s=1)
    ax1.set_title("Original Point Cloud")

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(remaining_points[:, 0], remaining_points[:, 1], remaining_points[:, 2], s=1)
    ax2.set_title("Incomplete Point Cloud")

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(remaining_points[:, 0], remaining_points[:, 1], remaining_points[:, 2], s=1, label='Remaining Points')
    ax3.scatter(removed_points[:, 0], removed_points[:, 1], removed_points[:, 2], s=1, c='r', label='Removed Points')
    ax3.set_title("Remaining vs Removed Points")
    ax3.legend()

    plt.show()

def visualize_two_PC(nodes_xyz_pre, nodes_xyz_gd, labels, filename='PC_recon.pdf'):
    color_dict = {0: '#BCB6AE', 1: '#BCB6AE', 2: '#BCB6AE'}
    colors = [color_dict[label] for label in labels]

    df_pre = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    df_gd = pd.DataFrame(nodes_xyz_gd, columns=['x', 'y', 'z'])

    fig = plt.figure(figsize=(4, 6))
    ax1 = fig.add_subplot(212, projection='3d')
    ax1.scatter(df_pre['x'], df_pre['y'], df_pre['z'], c=colors, s=1.5)  
    ax1.set_axis_off() # Hide coordinate space
    ax2 = fig.add_subplot(211, projection='3d')
    ax2.scatter(df_gd['x'], df_gd['y'], df_gd['z'], c=colors, s=1.5)    
    ax2.set_axis_off() # Hide coordinate space
    plt.subplots_adjust(hspace=0)
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)

def visualize_PC_with_label(nodes_xyz, labels=1, filename='PC_label.pdf'):
    # plot in 3d using plotly
    df = pd.DataFrame(nodes_xyz, columns=['x', 'y', 'z'])
    # define custom colors for each category
    # colors = {'0': '#BCB6AE', '1': '#288596', '3': '#7D9083'}
    # colors = {'0': 'grey', '1': 'blue', '3': 'red'}
    # df['color'] = label.astype(int)
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color = 'color', color_discrete_sequence=[colors[k] for k in sorted(colors.keys())])
    # # fig = px.scatter_3d(df, x='x', y='y', z='z', color = clr_nodes, color_continuous_scale=px.colors.sequential.Viridis)
    # fig.update_traces(marker_size = 1.5)  # increase marker_size for bigger node size
    # fig.show()   
    # plotly.offline.plot(fig)
    # fig.write_image(filename) 

    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}
    # color_dict = {0: '#BCB6AE', 1: '#288596'}
    colors = [color_dict[label] for label in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=colors, s=1.5)  
    ax.set_axis_off() # Hide coordinate space
    plt.savefig(filename)
    plt.close(fig)

def save_coord_for_visualization(data, savename):
    with open('./log/' + savename+'_LVendo.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 0]) + ',' + str(data[i, 1]) + ',' + str(data[i, 2]) + '\n')
    with open('./log/' + savename+'_epi.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 3]) + ',' + str(data[i, 4]) + ',' + str(data[i, 5]) + '\n')
    with open('./log/' + savename+'_RVendo.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 6]) + ',' + str(data[i, 7]) + ',' + str(data[i, 8]) + '\n')

def lossplot_detailed(lossfile_train, lossfile_val, lossfile_electrode_train, lossfile_electrode_val, lossfile_SSM_train, lossfile_SSM_val, lossfile_kp_train, lossfile_kp_val):
    ax = plt.subplot(221)
    ax.set_title('total loss')
    lossplot(lossfile_train, lossfile_val)

    ax = plt.subplot(222)
    ax.set_title('electrode loss')
    lossplot(lossfile_electrode_train, lossfile_electrode_val)

    ax = plt.subplot(223)
    ax.set_title('recon loss')
    lossplot(lossfile_SSM_train, lossfile_SSM_val)

    ax = plt.subplot(224)
    ax.set_title('keypoint loss')
    lossplot(lossfile_kp_train, lossfile_kp_val)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    plt.savefig("img.png")
    # plt.show()

def lossplot(lossfile1, lossfile2):

    loss = np.loadtxt(lossfile1)
    if loss.size == 0:
        print(f"Warning: No data to plot in file: {lossfile1}")
        return

    x = range(0, loss.size)
    y = loss
    plt.plot(x, y, '#FF7F61') # , label='train'
    plt.legend(frameon=False)

    loss = np.loadtxt(lossfile2)
    if loss.size == 0:
        print(f"Warning: No data to plot in file: {lossfile2}")
        return
    x = range(0, loss.size)
    y = loss
    plt.plot(x, y, '#2C4068') # , label='val'
    plt.legend(frameon=False)
    # plt.show()
    # plt.savefig("img.png")

def ECG_visual_two(prop_data, target_ecg):   
    prop_data[target_ecg[np.newaxis, ...] == 0.0], target_ecg[target_ecg == 0.0] = np.nan, np.nan

    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axs = plt.subplots(2, 8, constrained_layout=True, figsize=(40, 10))
    for i in range(8):
        leadName = leadNames[i]
        axs[0, i].plot(prop_data[0, i, :], color=[223/256,176/256,160/256], label='pred', linewidth=4)
        for j in range(1, prop_data.shape[0]):
            axs[0, i].plot(prop_data[j, i, :], color=[223/256,176/256,160/256], linewidth=4) 
        axs[0, i].plot(target_ecg[i, :], color=[154/256,181/256,174/256], label='true', linewidth=4)
        axs[0, i].set_title('Lead ' + leadName, fontsize=20)
        axs[0, i].set_axis_off() 
        axs[1, i].set_axis_off() 
    axs[0, i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    fig.savefig("ECG_visual.pdf")
    plt.show()
    plt.close(fig)

def ECG_visual_single_row(ecg_pre, ecg_gd, subject_name):
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axs = plt.subplots(1, 8, constrained_layout=True, figsize=(15, 3))  # Change the subplot shape
    for i in range(8):
        leadName = leadNames[i]
        axs[i].plot(ecg_pre[i, :], color='#91c1d5', linestyle='--', label='pred', linewidth=2)
        axs[i].plot(ecg_gd[i, :], color='#91c1d5', label='true', linewidth=2)
        axs[i].set_title('Lead ' + leadName, fontsize=12)  # Reduce title fontsize
        axs[i].set_axis_off() 

    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)  # Reduce legend fontsize
    fig.savefig('./results/' + subject_name + '_ECG_compare.pdf')
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    datapath = './dataset_cardiac_mesh'
    datafile = sorted(glob.glob(datapath + '/1*'))
    for subjectid in range(len(datafile)):
        subject_name = datafile[subjectid].replace(datapath, '').replace('\\', '')
        print(subject_name)

        predict_ecg_name = datafile[subjectid] + '/' + subject_name + '_simulated_ECG_normal_pre.csv'
        gd_ecg_name =datafile[subjectid] + '/' + subject_name + '_simulated_ECG_normal_gd.csv'
        ecg_pre = np.loadtxt(predict_ecg_name, delimiter=',')
        ecg_gd = np.loadtxt(gd_ecg_name, delimiter=',')
        ECG_visual_single_row(ecg_pre, ecg_gd, subject_name)
 