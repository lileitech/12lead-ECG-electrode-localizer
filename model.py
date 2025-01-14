from tkinter import Y
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.functional import relu

from models.pointnet_utils import PointNetEncoder
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from utils import init_graph, Batch_Keypoint_graph_interpolation_v2, farthest_feature_sample, gather_features

class Electrode_Net(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, in_ch=3, num_input=1024, n_electrodes=10, n_keypoints=20, latent_dim=1024, grid_size=2):
        super().__init__()

        self.num_dense = 2*num_input
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_keypoints*3)
        )

        self.kp_generator = MultiKPGenerator(latent_dim, n_keypoints)

        self.decoder = Multi_Offset_Predictor(num_input)

    def forward(self, pc_torso):
        B, _, N = pc_torso.shape
        xyz = pc_torso[:,:3,:] 
        
        # encoder
        feature = self.first_conv(xyz)                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)

        # decoder - electrodes
        # out_keypoints = self.classifier(feature_global)
        out_keypoints = self.kp_generator(feature, feature_global)
        
        # decoder - torso
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        
        pred_kp = out_keypoints#.reshape(B, -1, 3)
        pred_A_matric = init_graph(coarse, pred_kp)
        pred_kp_coarse1, pred_kp_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp, pred_A_matric)

        global_feat = feature_global
        pcd = torch.cat([coarse, pred_kp_coarse1], dim=1)

        fine = self.decoder(pcd, global_feat)

        visual_check = False
        if visual_check:
            elev_degree = 3
            azim_degree = -83
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(9, 12))
            ax = fig.add_subplot(221, projection='3d')
            out_electrode_batch = pred_kp[0].detach().cpu().numpy()
            pc_torso_batch = pred_kp_coarse1[0].detach().cpu().numpy()
            # ax.scatter(pc_torso_batch[:, 0], pc_torso_batch[:, 1], pc_torso_batch[:, 2], s=5, c = '#DAEFF2', label='sparse skeleton') 
            ax.scatter(out_electrode_batch[10:, 0], out_electrode_batch[10:, 1], out_electrode_batch[10:, 2], s=15, c='#29B4B6', label='keypoints') # C1A9A9
            ax.scatter(out_electrode_batch[:10, 0], out_electrode_batch[:10, 1], out_electrode_batch[:10, 2], s=15, c='#F0776d', label='electrode')   # #FDDA99
            ax.set_axis_off()
            # ax.legend()
            ax.view_init(elev=elev_degree, azim=azim_degree)

            ax = fig.add_subplot(222, projection='3d')
            # pc_torso_batch = pred_kp_coarse2[0].detach().cpu().numpy()
            ax.scatter(pc_torso_batch[:, 0], pc_torso_batch[:, 1], pc_torso_batch[:, 2], s=5, c = '#DAEFF2', label='surface skeleton') 
            ax.scatter(out_electrode_batch[10:, 0], out_electrode_batch[10:, 1], out_electrode_batch[10:, 2], s=15, c='#29B4B6', label='keypoints')
            ax.scatter(out_electrode_batch[:10, 0], out_electrode_batch[:10, 1], out_electrode_batch[:10, 2], s=15, c='#F0776d', label='electrode')  
            ax.set_axis_off()
            # ax.legend()
            ax.view_init(elev=elev_degree, azim=azim_degree)

            ax = fig.add_subplot(223, projection='3d')
            pc_torso_batch = coarse[0].detach().cpu().numpy()
            ax.scatter(pc_torso_batch[:, 0], pc_torso_batch[:, 1], pc_torso_batch[:, 2], s=5, c = '#91c1d5', label='coarse torso') 
            # ax.scatter(out_electrode_batch[:, 0], out_electrode_batch[:, 1], out_electrode_batch[:, 2], s=10, c='#B57775', label='keypoints') 
            ax.set_axis_off()
            # ax.legend()
            ax.view_init(elev=elev_degree, azim=azim_degree)

            ax = fig.add_subplot(224, projection='3d')
            pc_torso_batch = fine[0].detach().cpu().numpy()
            ax.scatter(pc_torso_batch[:, 0], pc_torso_batch[:, 1], pc_torso_batch[:, 2], s=5, c = '#91c1d5', label='dense torso') 
            # ax.scatter(out_electrode_batch[:, 0], out_electrode_batch[:, 1], out_electrode_batch[:, 2], s=10, c='#B57775', label='keypoints') 
            ax.set_axis_off()
            # ax.legend()
            ax.view_init(elev=elev_degree, azim=azim_degree)
    
            plt.savefig('fig_result_keypoint_skeleton.pdf')
            plt.show()

        return out_keypoints, coarse.contiguous(), fine.contiguous()
    
class Electrode_Net_pcn(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, in_ch=3, num_input=1024, n_electrodes=10, n_keypoints=20, latent_dim=1024, grid_size=2):
        super().__init__()

        self.num_dense = 2*num_input
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)  # (1, 2, S)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_electrodes*3)
        )
      
        # add uncertainty measurement for the electrodes

    def forward(self, pc_torso):
        B, _, N = pc_torso.shape
        xyz = pc_torso[:,:3,:] 
        
        # encoder
        feature = self.first_conv(xyz)                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]                           # (B, 1024)

        # decoder - electrodes
        electrodes = self.classifier(feature_global)
        
        # decoder - keypoints
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return electrodes.view(B, -1, 3), coarse.contiguous(), fine.transpose(1, 2).contiguous()
    
class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

def ffs(feature, num):
    # b, n_latent_feature, n_points
    pid_ffs = farthest_feature_sample(feature.transpose(1, 2).contiguous(), num)
    x_ffs = gather_features(feature.transpose(1, 2).contiguous(), pid_ffs)

    return x_ffs.transpose(1, 2).contiguous() # b, D, Nf

class MultiKPGenerator(nn.Module):
    def __init__(self, dim_feat=1024, num_kp1=64):
        super(MultiKPGenerator, self).__init__()
        
        self.num_kp1 = num_kp1

        self.ps1 = nn.ConvTranspose1d(dim_feat, 128, num_kp1, bias=True)
        # self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer added by Lei on 2024/07/22

        self.mlp_1 = MLP_Res(in_dim=dim_feat * 2 + 128, hidden_dim=128, out_dim=128)
        # self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer added by Lei on 2024/07/22
        self.mlp_2 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, point_feat, global_feat):
        """
        Args:
            point_feat: Tensor (b, 1024, 2048)
            global_feat: Tensor (b, 1024)
        """
        feat_ffs1 = ffs(point_feat, self.num_kp1) # (b, 1024, 64)

        global_feat = global_feat.unsqueeze(-1) # (b, 1024, 1)
        
        x1 = self.ps1(global_feat)  # (b, 128, 64)
        # x1 = self.dropout1(x1)  # Apply dropout added by Lei on 2024/07/22

        x1 = self.mlp_1(torch.cat([x1, feat_ffs1, global_feat.repeat((1, 1, feat_ffs1.size(2)))], 1)) # (b, 128, 64)
        # x1 = self.dropout2(x1)  # Apply dropout added by Lei on 2024/07/22

        pred_kp1 = self.mlp_2(x1) # (b, 3, 64)

        return pred_kp1.transpose(1,2).contiguous()

class Multi_Offset_Predictor(nn.Module):
    def __init__(self, num_input=1024, up_factor=2):
        super(Multi_Offset_Predictor, self).__init__()
        self.num_dense = up_factor*num_input
        self.grid_size = up_factor
        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, pcd_prev, global_feat):
        # pcd_prev: (B, 2048, 3), gloal_feat: (B, N_feat)
        B, _, N = pcd_prev.shape
        global_feat = global_feat.unsqueeze(2)
        point_feat = pcd_prev.unsqueeze(2).expand(-1, -1, self.grid_size, -1)             # (B, num_coarse*2, 4, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = global_feat.expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine_pcd = self.final_conv(feat) + point_feat 

        return fine_pcd.transpose(1, 2).contiguous()
  
class Multi_Offset_Predictor_old(nn.Module):
    def __init__(self, dim_input=1024, dim_output=256, up_factor=2, i=0, radius=1):
        super(Multi_Offset_Predictor, self).__init__()
        self.up_factor = up_factor
        self.i = i
        self.radius = radius

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_input, layer_dims=[256, dim_output])
        self.mlp_res = MLP_Res(in_dim=dim_output, hidden_dim=64, out_dim=dim_output)
        self.mlp_3 = MLP_CONV(in_channel=dim_output, layer_dims=[dim_output//2, 64, 3])

        # self.ps = nn.ConvTranspose1d(128, dim_output, up_factor, up_factor, bias=False)   # point-wise splitting

    def forward(self, pcd_prev, global_feat):
        # pcd_prev: (B, 2048, 3), gloal_feat: (B, N_feat)
        num_point = pcd_prev.shape[1]
        global_feat = global_feat.unsqueeze(2)
        pcd_up = pcd_prev.repeat(1, self.up_factor, 1)

        feat_prev = self.mlp_1(pcd_up.transpose(1,2).contiguous()) # [B, 128, N]
        feat_1 = torch.cat([feat_prev, torch.max(feat_prev, 2, keepdim=True)[0].repeat((1, 1, num_point * self.up_factor)), global_feat.repeat(1, 1, num_point*self.up_factor)], 1)
        feat_up = self.mlp_2(feat_1) # [B, 128, 2048]
        # feat_up = self.ps(feat_2)
        curr_feat = self.mlp_res(feat_up)
        delta = torch.tanh(self.mlp_3(F.relu(curr_feat))) / self.radius**self.i
        fine_pcd = pcd_prev.repeat(1, self.up_factor, 1) + delta.transpose(1,2).contiguous()  # or upsample [B, N * up_factor, 3]

        return fine_pcd
    
def point_maxpool(features, npts, keepdim=True):
    splitted = torch.split(features, npts[0], dim=1)
    outputs = [torch.max(f, dim=2, keepdim=keepdim)[0] for f in splitted] # modified by Lei in 2022/02/10
    return torch.cat(outputs, dim=0)
    # return torch.max(features, dim=2, keepdims=keepdims)[0]

def point_unpool(features, npts):
    features = torch.split(features, features.shape[0], dim=0)
    outputs = [f.repeat(1, 1, npts[i]) for i, f in enumerate(features)]
    # outputs = [torch.tile(f, [1, 1, npts[i]]) for i, f in enumerate(features)]
    return torch.cat(outputs, dim=0)
    # return features.repeat([1, 1, 256])

class mlp_conv(nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(mlp_conv, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(self.layer_dims):
            layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            setattr(self, 'conv_' + str(i), layer)
            in_channels = out_channels

    def __call__(self, inputs):
        outputs = inputs
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'conv_' + str(i))
            if i == dims - 1:
                outputs = layer(outputs)
            else:
                outputs = relu(layer(outputs))
        return outputs

class mlp(nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(mlp, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(layer_dims):
            layer = torch.nn.Linear(in_channels, out_channels)
            setattr(self, 'fc_' + str(i), layer)
            in_channels = out_channels

    def __call__(self, inputs):
        outputs = inputs
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'fc_' + str(i))
            if i == dims - 1:
                outputs = layer(outputs)
            else:
                outputs = relu(layer(outputs))
        return outputs

class DeepSSM_Net(nn.Module):
    def __init__(self, out_ssm, n_electrode = 10):
        super(DeepSSM_Net, self).__init__()

        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(npoint=n_electrode, radius=0.2, nsample=8, in_channel=3, mlp=[8, 8, 16], group_all=False)
        # self.sa2 = PointNetSetAbstraction(16, 0.4, 8, 16 + 3, [16, 16, 32], False)

        # PointNet++ Decoder
        self.fc1 = nn.Linear(10*16, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, out_ssm)

    def forward(self, pc_electrode):

        # extract point cloud features      
        l0_xyz = pc_electrode[:,:3,:] 
        l0_points = None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        features = l1_points.view(-1, 10*16)

        out = self.drop1(F.relu(self.bn1(self.fc1(features))))
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    x = torch.rand(3, 4, 2048)
    conditions = torch.rand(3, 2, 1)

    network = Electrode_Net()
    y_coarse, y_detail = network(x, conditions)
    print(y_coarse.size(), y_detail.size())
