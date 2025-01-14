import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from dataset import LoadDataset
from model import Electrode_Net
from utils import chamfer_distance, plot_two_pcd_with_edge, plot_pcd
from loss import F_loss_electrode, F_loss_recon

elev_degree = 3
azim_degree = -83

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset_mesh')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--model', type=str, default='log/net_model.pkl') #'log/net_model.pkl'
    parser.add_argument('--in_ch', type=int, default=3) # coordinate dimension + label index
    parser.add_argument('--n_electrodes', type=int, default=10) 
    parser.add_argument('--n_keypoints', type=int, default=128) 
    parser.add_argument('--out_ssm', type=int, default=30+16)
    parser.add_argument('--num_input', type=int, default=1024*2) 
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lamda', type=float, default=0.01) 
    parser.add_argument('--base_lr', type=float, default=5e-5) 
    parser.add_argument('--lr_decay_steps', type=int, default=50) 
    parser.add_argument('--lr_decay_rate', type=float, default=0.5) 
    parser.add_argument('--weight_decay', type=float, default=1e-3) 
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--load_SSM', type=bool, default=False)
    args = parser.parse_args()

    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = LoadDataset(path=args.data_root, num_input=args.num_input, num_kp=args.n_keypoints, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    network = Electrode_Net(in_ch=args.in_ch, num_input=args.num_input, n_electrodes=args.n_electrodes, n_keypoints=args.n_keypoints)

    network.load_state_dict(torch.load('log/net_model.pkl', map_location=torch.device('cpu')))
    network.to(DEVICE)

    all_results =  {'Torso_Reconstruction_Error': [], 'Electrode_Localization_Error': []}
    with open('./my_split/test.list', 'r') as f:
        filenames = [line.strip() for line in f]

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 1):
            pc_torso, gt_electrode, mesh_torso_coarse, mesh_torso_dense, mesh_torso_kp = data
            pc_torso, gt_electrode = pc_torso.to(DEVICE), gt_electrode.to(DEVICE) 
            mesh_torso_coarse, mesh_torso_dense = mesh_torso_coarse.to(DEVICE), mesh_torso_dense.to(DEVICE) 
            mesh_torso_kp = mesh_torso_kp.to(DEVICE)
            pc_torso = pc_torso.permute(0, 2, 1)   
            
            out_keypoint, y_coarse, y_detail = network(pc_torso) 

            loss_electrode, loss_keypoint = F_loss_electrode(out_keypoint, gt_electrode, mesh_torso_coarse)   

            loss_SSM = F_loss_recon(pc_torso, y_coarse, y_detail, mesh_torso_coarse, mesh_torso_dense)

            out_electrode_batch = out_keypoint[:, :args.n_electrodes][0].detach().cpu().numpy()
            gt_electrode_batch = gt_electrode[0].detach().cpu().numpy()
            y_detail_batch = y_detail[0].detach().cpu().numpy()
            mesh_torso_dense_batch = mesh_torso_dense[0].detach().cpu().numpy()

            save_prediction = False
            if save_prediction:
                save_dir = args.save_fold + '/'+ filenames[i-1]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.savetxt(save_dir + '/'+ 'electrodes_TIM_64kp_30rr_pre_v02.csv', out_electrode_batch, fmt='%1.8f', delimiter=',')

            chamfer_dist = chamfer_distance(y_detail_batch, mesh_torso_dense_batch)
            print('Torso reconstruction error: ' + str(chamfer_dist))
            individual_dist = np.linalg.norm(out_electrode_batch - gt_electrode_batch, axis=1)
            euclidean_dist = np.mean(individual_dist)
            print('Electrode localization error: ' + str(euclidean_dist))

            all_results['Torso_Reconstruction_Error'].append(chamfer_dist)
            all_results['Electrode_Localization_Error'].append(euclidean_dist)

            # # Calculate the distances between corresponding points
            # source_points, target_points = y_detail_batch, mesh_torso_dense_batch
            # # source_points, target_points = out_electrode_batch, gt_electrode_batch  
            # distances = np.linalg.norm(source_points - target_points, axis=1)
            # # Visualize the error map
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # sc = ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], s=5, c=distances, cmap='viridis', marker='o')
            # # sc = ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c=distances, cmap='plasma', marker='o')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # fig.colorbar(sc, ax=ax, label='Distance')
            # plt.show()

            visual_check = False
            if visual_check:                   
                y_coarse_batch = y_coarse[0].detach().cpu().numpy()
                y_detail_batch = y_detail[0].detach().cpu().numpy()
                pc_torso_batch = mesh_torso_kp.permute(0, 2, 1)[0].detach().cpu().numpy()
                mesh_torso_dense_batch = mesh_torso_dense[0].detach().cpu().numpy()

                fig = plt.figure(figsize=(12, 8))

                ax = fig.add_subplot(221, projection='3d')
                
                ax.scatter(pc_torso_batch[0, :], pc_torso_batch[1, :], pc_torso_batch[2, :], s=5, c = '#91c1d5', label='partial torso') 
                # ax.scatter(gt_electrode_batch[:, 0], gt_electrode_batch[:, 1], gt_electrode_batch[:, 2], s=10, c='#edc162', label='electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)  # Rotate 25 degrees azimuth
            
                ax = fig.add_subplot(222, projection='3d')
                plot_pcd(ax, out_electrode_batch)
                ax.view_init(elev=elev_degree, azim=azim_degree)  # Rotate 25 degrees azimuth

                ax = fig.add_subplot(223, projection='3d')
                ax.scatter(y_coarse_batch[:, 0], y_coarse_batch[:, 1], y_coarse_batch[:, 2], s=2, c = '#91c1d5', label='gd dense torso') 
                ax.set_axis_off()
                ax.view_init(elev=elev_degree, azim=azim_degree)  # Rotate 25 degrees azimuth

                ax = fig.add_subplot(224, projection='3d')
                ax.scatter(y_detail_batch[:, 0], y_detail_batch[:, 1], y_detail_batch[:, 2], s=2, c = '#91c1d5', label='pre dense torso')
                # ax.scatter(gt_electrode_batch[:, 0], gt_electrode_batch[:, 1], gt_electrode_batch[:, 2], s=10, c='#edc162', label='gd electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)  # Rotate 25 degrees azimuth

                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)  # Adjust space between plots
                                                
                plt.savefig('result_visual.pdf')
                plt.show()

            visual_check = False
            if visual_check:          
                y_coarse_batch = y_coarse[0].detach().cpu().numpy()
                y_detail_batch = y_detail[0].detach().cpu().numpy()

                fig = plt.figure(figsize=(10, 18))
                ax = fig.add_subplot(321, projection='3d')
                plot_two_pcd_with_edge(ax, out_electrode_batch, gt_electrode_batch)
                ax.view_init(elev=elev_degree, azim=azim_degree)

                ax = fig.add_subplot(322, projection='3d')
                pc_torso_batch = pc_torso[0].detach().cpu().numpy()
                ax.scatter(pc_torso_batch[0, :], pc_torso_batch[1, :], pc_torso_batch[2, :], s=5, c = '#91c1d5', label='partial torso') 
                # ax.scatter(gt_electrode_batch[:, 0], gt_electrode_batch[:, 1], gt_electrode_batch[:, 2], s=10, c='#edc162', label='electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)

                ax = fig.add_subplot(323, projection='3d')
                mesh_torso_coarse_batch = mesh_torso_coarse[0].detach().cpu().numpy()
                ax.scatter(mesh_torso_coarse_batch[:, 0], mesh_torso_coarse_batch[:, 1], mesh_torso_coarse_batch[:, 2], s=5, c = '#91c1d5', label='gd coarse torso') 
                ax.scatter(y_coarse_batch[:, 0], y_coarse_batch[:, 1], y_coarse_batch[:, 2], s=5, c = '#edc162', label='pre coarse torso')
                # ax.scatter(gt_electrode_batch[:, 0], gt_electrode_batch[:, 1], gt_electrode_batch[:, 2], s=10, c='#edc162', label='gd electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)

                ax = fig.add_subplot(324, projection='3d')
                mesh_torso_dense_batch = mesh_torso_dense[0].detach().cpu().numpy()
                ax.scatter(mesh_torso_dense_batch[:, 0], mesh_torso_dense_batch[:, 1], mesh_torso_dense_batch[:, 2], s=2, c = '#91c1d5', label='gd dense torso') 
                ax.scatter(y_detail_batch[:, 0], y_detail_batch[:, 1], y_detail_batch[:, 2], s=2, c = '#edc162', label='pre dense torso')
                # ax.scatter(gt_electrode_batch[:, 0], gt_electrode_batch[:, 1], gt_electrode_batch[:, 2], s=10, c='#edc162', label='gd electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)

                ax = fig.add_subplot(325, projection='3d')               
                ax.scatter(y_coarse_batch[:, 0], y_coarse_batch[:, 1], y_coarse_batch[:, 2], s=5, c = '#91c1d5', label='pre coarse torso') 
                # ax.scatter(out_electrode_batch[:, 0], out_electrode_batch[:, 1], out_electrode_batch[:, 2], s=10, c='#edc162', label='pre electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)

                ax = fig.add_subplot(326, projection='3d')               
                ax.scatter(y_detail_batch[:, 0], y_detail_batch[:, 1], y_detail_batch[:, 2], s=2, c = '#91c1d5', label='pre dense torso') 
                # ax.scatter(out_electrode_batch[:, 0], out_electrode_batch[:, 1], out_electrode_batch[:, 2], s=10, c='#edc162', label='pre electrodes') 
                ax.set_axis_off()
                # ax.legend()
                ax.view_init(elev=elev_degree, azim=azim_degree)

                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)  # Adjust space between plots
                                                
                plt.savefig('fig_result_input_output.pdf')
                plt.show()

        errors_df = pd.DataFrame(all_results)
        errors_df.to_csv('./results/results.csv', index=False)
        print('Lei, well done!')       



