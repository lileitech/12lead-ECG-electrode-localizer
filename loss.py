import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from utils import plot_two_pcd_with_edge
# from pytorch3d.loss import chamfer_distance
from sklearn.mixture import GaussianMixture

def differentiable_volume_approximation_batched(pc, sigma=0.1):
    # points is expected to have the shape [n_batch, 3, n_points]
    _, n_points, _ = pc.shape  # [n_batch, n_points, 3]
    
    distances = torch.cdist(pc, pc)  # [n_batch, n_points, n_points]
    
    # Calculate Gaussian influence for each pair of points
    gaussians = torch.exp(-distances ** 2 / (2 * sigma ** 2))
    
    # Sum Gaussian influences to approximate volume
    volume_approx = gaussians.sum(dim=(1, 2)) / (n_points ** 2)
    
    return volume_approx

def calculate_volume_difference(pc1, pc2):
    volume1 = differentiable_volume_approximation_batched(pc1)
    volume2 = differentiable_volume_approximation_batched(pc2)
    
    # Calculate volume difference
    volume_diff = torch.abs(volume1 - volume2)
    
    return torch.mean(volume_diff)

def F_loss(out_electrode, out_b_affine, gt_electrode, SSM_vector):
    loss_electrode = F_loss_electrode(out_electrode, gt_electrode)
    loss_SSM = F_loss_SSM(out_electrode, out_b_affine, SSM_vector)

    return loss_electrode, loss_SSM

def F_loss_SSM(out_electrode, out_b_affine, SSM_vector, n_electrode = 10, visual_check=True):
    
    n_batch = out_electrode.shape[0]
    Pvec, Pval, x_bar = SSM_vector[0].unsqueeze(0).expand(n_batch, -1, -1), SSM_vector[1].unsqueeze(0).expand(n_batch, -1), SSM_vector[2].unsqueeze(0).expand(n_batch, -1)

    out_b, out_affine = out_b_affine[:, 0:Pval.shape[1]], out_b_affine[:, Pval.shape[1]:]
    out_b_new = 3*nn.Tanh()(out_b)*torch.sqrt(Pval)
    SSM_electrode = x_bar.unsqueeze(-1) + torch.matmul(Pvec, out_b_new.unsqueeze(-1))
    SSM_electrode_new = SSM_electrode.view(n_batch, n_electrode, 3)
    out_b_affine_new = out_affine.view(n_batch, 4, 4)
    SSM_electrode_alligned = F_batch_affine3d(SSM_electrode_new, out_b_affine_new) 
    lossfunc1 = nn.L1Loss()
    out_electrode = out_electrode.permute(0, 2, 1) 
    loss_SSM = lossfunc1(SSM_electrode_alligned, out_electrode)

    if visual_check:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        # SSM_electrode_new_batch = SSM_electrode_new[0].detach().cpu().numpy()
        SSM_electrode_alligned_batch = SSM_electrode_alligned[0].detach().cpu().numpy()
        out_electrode_batch = out_electrode[0].detach().cpu().numpy()
        plot_two_pcd_with_edge(ax, SSM_electrode_alligned_batch, out_electrode_batch)
        plt.show()
        # plt.savefig('result_SSM.png')

    return loss_SSM

def calculate_chamfer_distance(x, y):
    dist_x_y = torch.cdist(x, y)
    min_dist_x_y, _ = torch.min(dist_x_y, dim=1)
    min_dist_y_x, _ = torch.min(dist_x_y, dim=0)
    chamfer_distance_loss = torch.mean(torch.mean(min_dist_x_y) + torch.mean(min_dist_y_x))
    # chamfer_distance_loss, _ = chamfer_distance(x, y) # Not compiled with GPU support

    return chamfer_distance_loss

def calculate_earth_mover_distance(point_cloud1, point_cloud2):

    # Compute pairwise distances between points in the point clouds
    distance_matrix = torch.cdist(point_cloud1, point_cloud2)

    # Normalize the distance matrix to form a probability matrix
    probability_matrix = F.softmax(-distance_matrix, dim=1)

    # Compute the Earth Mover's Distance
    emd_loss = torch.mean(distance_matrix * probability_matrix)

    return emd_loss

def F_loss_recon(pc_torso, y_coarse, y_detail, coarse_gt, dense_gt, alpha = 100, beta=5):
    loss_coarse_cf = calculate_chamfer_distance(y_coarse, coarse_gt) 
    loss_fine_cf = calculate_chamfer_distance(y_detail, dense_gt) 
    # loss_fine_emd = calculate_earth_mover_distance(y_detail, dense_gt)
    # loss_coarse_emd = calculate_earth_mover_distance(y_coarse, coarse_gt)
    # loss_coarse = loss_coarse_cf + alpha*loss_coarse_emd
    # loss_fine = loss_fine_cf + alpha*loss_fine_emd

    loss_soft_shape = loss_coarse_cf + beta*loss_fine_cf

    loss_volume = calculate_volume_difference(y_coarse, coarse_gt) + beta*calculate_volume_difference(y_detail, dense_gt)

    # pc_torso = pc_torso.permute(0, 2, 1)
    # loss_input = calculate_chamfer_distance(y_coarse, pc_torso[:, :, :3]) + beta*calculate_chamfer_distance(y_detail, pc_torso[:, :, :3]) 
    

    # lossfunc1 = nn.L1Loss()
    # loss_hard_shape = lossfunc1(y_coarse, coarse_gt) + beta*lossfunc1(y_detail, dense_gt)

    return loss_soft_shape + 10*loss_volume # + loss_hard_shape

def F_loss_electrode(out_keypoints, gt_electrode, kp_gt, n_electrodes=10, alpha=0):
    n_batch = out_keypoints.shape[0]
    out_electrode = out_keypoints[:, :n_electrodes]
    output_electrode = out_electrode #.view(n_batch, n_electrodes, 3)
    # output_other_keypoints = out_keypoints[:, n_electrodes*3:].view(n_batch, -1, 3)
    
    lossfunc1 = nn.L1Loss()
    loss_electrode = lossfunc1(output_electrode, gt_electrode)
    
    output_keypoints = out_keypoints #[:, n_electrodes:] #.view(n_batch, -1, 3)
    loss_keypoint = calculate_chamfer_distance(output_keypoints, kp_gt)
    
    # + alpha*calculate_earth_mover_distance(output_keypoints, kp_gt)

    # n_keypoints = output_keypoints.shape[1]
    # pairwise_distances  = torch.norm(output_keypoints[:, :, None, :] - output_keypoints[:, None, :, :], dim=-1)
    # mask = ~torch.eye(n_keypoints, dtype=torch.bool, device=out_keypoints.device)
    # dispersion_loss = torch.mean(pairwise_distances[:, mask])
    # spatial_term = torch.mean(torch.pow(F.pairwise_distance(output_keypoints.unsqueeze(1), output_keypoints.unsqueeze(0)), 2))
    # unique_term = -torch.mean(torch.abs(F.pairwise_distance(output_keypoints.unsqueeze(1), output_keypoints.unsqueeze(0))))

    return loss_electrode, loss_keypoint # + 0.1*(dispersion_loss + spatial_term + unique_term)

def F_batch_affine3d(input, matrix, max_rotation_deg=10.0, max_scaling_factor=1.2, min_scaling_factor=0.8):
    """
    input : torch.Tensor
        shape = (n_batch, n_points, 3)
    matrix : torch.Tensor
        shape = (n_batch, 4, 4)
    max_rotation_deg : float, optional
        Maximum rotation angle in degrees (default is 10.0).
    max_scaling_factor : float, optional
        Maximum scaling factor (default is 1.2).
    min_scaling_factor : float, optional
        Minimum scaling factor (default is 0.6).
    """
    A_batch = matrix[:, :3, :3]
    b_batch = matrix[:, :3, 3].unsqueeze(1)
    device = matrix.device
    max_rotation_angle = torch.tensor(max_rotation_deg, device=device)

    # Apply rotation constraints
    u, _, v = A_batch.svd()
    rotation_matrix = torch.matmul(u, v.transpose(2, 1))

    # Extract the scaling factors
    scaling_factors = torch.norm(A_batch, dim=2)
    scaling_factors = torch.abs(scaling_factors)

    # Apply rotation angle constraint
    rotation_angles = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
    rotation_angles_clamp = torch.clamp(rotation_angles, -torch.deg2rad(max_rotation_angle),
                                  torch.deg2rad(max_rotation_angle))
    rotation_matrix_clamp = torch.stack([
        torch.cos(rotation_angles_clamp), -torch.sin(rotation_angles_clamp), torch.zeros_like(rotation_angles_clamp),
        torch.sin(rotation_angles_clamp), torch.cos(rotation_angles_clamp), torch.zeros_like(rotation_angles_clamp),
        torch.zeros_like(rotation_angles_clamp), torch.zeros_like(rotation_angles_clamp), torch.ones_like(rotation_angles_clamp)
    ], dim=1).view(-1, 3, 3)

    # Apply scaling factor constraints
    scaling_factors_clamp = torch.clamp(scaling_factors, torch.tensor(min_scaling_factor, device=device), torch.tensor(max_scaling_factor, device=device))

    scaled_rotation_matrix_clamp = rotation_matrix_clamp
    scaled_rotation_matrix_clamp *= scaling_factors_clamp.unsqueeze(-1)

    # Apply the coordinate transformation
    new_coords = input.bmm(scaled_rotation_matrix_clamp.transpose(1, 2)) + b_batch.expand_as(input)

    return new_coords

class AffineCOMTransform(nn.Module):
    def __init__(self, use_com=True):
        super(AffineCOMTransform, self).__init__()

        self.translation_m = None
        self.rotation_x = None
        self.rotation_y = None
        self.rotation_z = None
        self.rotation_m = None
        self.shearing_m = None
        self.scaling_m = None

        self.id = torch.zeros((1, 3, 4)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

        self.use_com = use_com

    def forward(self, point_cloud, affine_para):
        # Matrix that registers x to its center of mass
        id_grid = F.affine_grid(self.id, point_cloud.shape, align_corners=True)

        to_center_matrix = torch.eye(4).cuda()
        reversed_to_center_matrix = torch.eye(4).cuda()
        if self.use_com:
            point_cloud_sum = torch.sum(point_cloud, dim=1, keepdim=True)
            center_mass = torch.sum(point_cloud * id_grid, dim=1, keepdim=True) / point_cloud_sum

            to_center_matrix[:, :3, 3] = center_mass.squeeze(1)
            reversed_to_center_matrix[:, :3, 3] = -center_mass.squeeze(1)

        self.translation_m = torch.eye(4).cuda()
        self.rotation_x = torch.eye(4).cuda()
        self.rotation_y = torch.eye(4).cuda()
        self.rotation_z = torch.eye(4).cuda()
        self.rotation_m = torch.eye(4).cuda()
        self.shearing_m = torch.eye(4).cuda()
        self.scaling_m = torch.eye(4).cuda()

        trans_xyz = affine_para[:, 0:3]
        rotate_xyz = affine_para[:, 3:6] * math.pi
        shearing_xyz = affine_para[:, 6:9] * math.pi
        scaling_xyz = 1 + (affine_para[:, 9:12] * 0.5)

        self.translation_m[:, :3, 3] = trans_xyz
        self.scaling_m[:, 0, 0] = scaling_xyz[:, 0]
        self.scaling_m[:, 1, 1] = scaling_xyz[:, 1]
        self.scaling_m[:, 2, 2] = scaling_xyz[:, 2]

        self.rotation_x[:, 1, 1] = torch.cos(rotate_xyz[:, 0])
        self.rotation_x[:, 1, 2] = -torch.sin(rotate_xyz[:, 0])
        self.rotation_x[:, 2, 1] = torch.sin(rotate_xyz[:, 0])
        self.rotation_x[:, 2, 2] = torch.cos(rotate_xyz[:, 0])

        self.rotation_y[:, 0, 0] = torch.cos(rotate_xyz[:, 1])
        self.rotation_y[:, 0, 2] = torch.sin(rotate_xyz[:, 1])
        self.rotation_y[:, 2, 0] = -torch.sin(rotate_xyz[:, 1])
        self.rotation_y[:, 2, 2] = torch.cos(rotate_xyz[:, 1])

        self.rotation_z[:, 0, 0] = torch.cos(rotate_xyz[:, 2])
        self.rotation_z[:, 0, 1] = -torch.sin(rotate_xyz[:, 2])
        self.rotation_z[:, 1, 0] = torch.sin(rotate_xyz[:, 2])
        self.rotation_z[:, 1, 1] = torch.cos(rotate_xyz[:, 2])

        self.rotation_m = torch.bmm(torch.bmm(self.rotation_z, self.rotation_y), self.rotation_x)

        self.shearing_m[:, 0, 1] = shearing_xyz[:, 0]
        self.shearing_m[:, 0, 2] = shearing_xyz[:, 1]
        self.shearing_m[:, 1, 2] = shearing_xyz[:, 2]

        output_affine_m = torch.bmm(torch.bmm(to_center_matrix, torch.bmm(self.shearing_m, torch.bmm(self.scaling_m,
                                                                                                       torch.bmm(
                                                                                                           self.rotation_m,
                                                                                                           torch.bmm(
                                                                                                               reversed_to_center_matrix,
                                                                                                               self.translation_m))))),
                                    id_grid.permute(0, 3, 1, 2))

        grid = F.affine_grid(output_affine_m[:, :3, :3], point_cloud.shape, align_corners=True)
        transformed_point_cloud = F.grid_sample(point_cloud, grid, mode='bilinear', align_corners=True)

        return transformed_point_cloud, output_affine_m[:, :3, :3]

def F_batch_affine3d_ori(input, matrix):
    """
    input : torch.Tensor
        shape = (n_batch, n_points, 3)
    matrix : torch.Tensor
        shape = (n_batch, 4, 4)
    """
    A_batch = matrix[:, :3, :3]
    b_batch = matrix[:, :3, 3].unsqueeze(1)

    # apply the coordinate transformation
    new_coords = input.bmm(A_batch.transpose(1, 2)) + b_batch.expand_as(input)

    return new_coords

def F_calculate_transmat(theta_x, theta_y, theta_z, scale_x, scale_y, scale_z, translation):
    # Convert angles to radians
    theta_x, theta_y, theta_z = torch.radians(theta_x), torch.radians(theta_y), torch.radians(theta_z)
    
    # Rotation matrices around each axis
    rotation_x = torch.tensor([[1, 0, 0],
                               [0, torch.cos(theta_x), -torch.sin(theta_x)],
                               [0, torch.sin(theta_x), torch.cos(theta_x)]])
    
    rotation_y = torch.tensor([[torch.cos(theta_y), 0, torch.sin(theta_y)],
                               [0, 1, 0],
                               [-torch.sin(theta_y), 0, torch.cos(theta_y)]])
    
    rotation_z = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                               [torch.sin(theta_z), torch.cos(theta_z), 0],
                               [0, 0, 1]])
    
    # Diagonal matrix for scaling
    scale_matrix = torch.tensor([[scale_x, 0, 0],
                                [0, scale_y, 0],
                                [0, 0, scale_z]])
    
    # Translation vector
    translation_vector = torch.tensor([[translation[0]],
                                       [translation[1]],
                                       [translation[2]]])
    
    # Combine transformations
    transformation_matrix = translation_vector + rotation_z @ rotation_y @ rotation_x @ scale_matrix
    
    return transformation_matrix


if __name__ == '__main__':

    pcs1 = torch.rand(10, 1024, 4)
    pcs2 = torch.rand(10, 1024, 4)



