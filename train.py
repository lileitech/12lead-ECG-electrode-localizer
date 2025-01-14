import argparse
import torch
torch.cuda.empty_cache() # clearing the occupied cuda memory
from torch.backends import cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import warnings
# Ignore the specific warning about the image Python extension
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension*")

from dataset import LoadDataset
from model import Electrode_Net
from loss import F_loss_recon, F_loss, F_loss_electrode, F_loss_recon
from utils import lossplot, lossplot_detailed, plot_two_pcd, plot_two_pcd_with_edge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset_mesh')
    parser.add_argument('--ssm_model', type=str, default='./results/ssm_obj.pkl')
    parser.add_argument('--model', type=str, default=None) #'log/net_model.pkl'
    parser.add_argument('--in_ch', type=int, default=3) # coordinate dimension + label index
    parser.add_argument('--n_electrodes', type=int, default=10) 
    parser.add_argument('--n_keypoints', type=int, default=128) # 32
    parser.add_argument('--out_ssm', type=int, default=30+16)
    parser.add_argument('--num_input', type=int, default=1024*2) 
    parser.add_argument('--batch_size', type=int, default=6) # 5
    parser.add_argument('--lamda', type=float, default=0.05) # 0.05
    parser.add_argument('--base_lr', type=float, default=1e-4) # 5e-5, 1e-4
    parser.add_argument('--lr_decay_steps', type=int, default=30) # 50
    parser.add_argument('--lr_decay_rate', type=float, default=0.5) 
    parser.add_argument('--weight_decay', type=float, default=1e-3) 
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--load_SSM', type=bool, default=False)
    args = parser.parse_args()

    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    
    if args.load_SSM:
        # Load the SSM object
        with open(args.ssm_model, 'rb') as file:
            ssm_obj = pickle.load(file)
        mean_shape_columnvector = ssm_obj.compute_dataset_mean()
        # mean_shape = mean_shape_columnvector.reshape(-1, 3) # x_bar
        shape_model_components = ssm_obj.pca_model_components # Pvec
        shape_parameter_vector = ssm_obj.model_parameters # Pval
        tensor_shape_model_components, tensor_shape_parameter_vector, tensor_mean_shape_columnvector = torch.from_numpy(shape_model_components).float().to(DEVICE), torch.from_numpy(shape_parameter_vector).float().to(DEVICE), torch.from_numpy(mean_shape_columnvector).float().to(DEVICE)
        SSM_vector = [tensor_shape_model_components, tensor_shape_parameter_vector, tensor_mean_shape_columnvector]

    train_dataset = LoadDataset(path=args.data_root, num_input=args.num_input, num_kp=args.n_keypoints, split='train')
    val_dataset = LoadDataset(path=args.data_root, num_input=args.num_input, num_kp=args.n_keypoints, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cudnn.benchmark = True

    network = Electrode_Net(in_ch=args.in_ch, num_input=args.num_input, n_electrodes=args.n_electrodes, n_keypoints=args.n_keypoints)

    if args.model is not None:
        print('Loaded trained model from {}.'.format(args.model))
        network.load_state_dict(torch.load(args.model))
    else:
        print('Begin training new model.')

    network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)

    max_iter = int(len(train_dataset) / args.batch_size + 0.5)
    minimum_loss = 1e4
    best_epoch = 0

    lossfile_train = args.log_dir + "/training_loss.txt"
    lossfile_val = args.log_dir + "/val_loss.txt"
    lossfile_electrode_train = args.log_dir + "/training_electrode_loss.txt"
    lossfile_electrode_val = args.log_dir + "/val_electrode_loss.txt"
    lossfile_recon_train = args.log_dir + "/training_recon_loss.txt"
    lossfile_recon_val = args.log_dir + "/val_recon_loss.txt"
    lossfile_kp_train = args.log_dir + "/training_kp_loss.txt"
    lossfile_kp_val = args.log_dir + "/val_kp_loss.txt"
    
    write_type = 'a' # a: additional writing; w: overwrite writing
    f_train = open(lossfile_train, write_type)
    f_val = open(lossfile_val, write_type)
    f_kp_train = open(lossfile_kp_train, write_type)
    f_kp_val = open(lossfile_kp_val, write_type)
    f_electrode_train = open(lossfile_electrode_train, write_type)
    f_electrode_val = open(lossfile_electrode_val, write_type)
    f_recon_train = open(lossfile_recon_train, write_type)
    f_recon_val = open(lossfile_recon_val, write_type)

    for epoch in range(1, args.epochs + 1):

        if ((epoch % 10) == 0) and (epoch != 0):  
            lossplot_detailed(lossfile_train, lossfile_val, lossfile_electrode_train, lossfile_electrode_val, lossfile_recon_train, lossfile_recon_val, lossfile_kp_train, lossfile_kp_val)

        # training
        network.train()
        total_loss, total_loss_electrode, total_loss_recon, total_loss_kp, iter_count = 0, 0, 0, 0, 0
        for i, data in enumerate(train_dataloader, 1):
            pc_torso, gt_electrode, mesh_torso_coarse, mesh_torso_dense, mesh_torso_kp = data
            pc_torso, gt_electrode = pc_torso.to(DEVICE), gt_electrode.to(DEVICE) 
            mesh_torso_coarse, mesh_torso_dense = mesh_torso_coarse.to(DEVICE), mesh_torso_dense.to(DEVICE) 
            mesh_torso_kp = mesh_torso_kp.to(DEVICE) 
            pc_torso = pc_torso.permute(0, 2, 1)     

            optimizer.zero_grad() 

            out_keypoint, y_coarse, y_detail = network(pc_torso)
            loss_electrode, loss_keypoints = F_loss_electrode(out_keypoint, gt_electrode, mesh_torso_kp)
            loss_recon = F_loss_recon(pc_torso, y_coarse, y_detail, mesh_torso_coarse, mesh_torso_dense)
            loss = loss_electrode + args.lamda*(loss_recon + loss_keypoints)

            loss.backward()
            optimizer.step()
        
            iter_count += 1
            total_loss += loss.item()
            total_loss_electrode += loss_electrode.item()
            total_loss_recon += loss_recon.item()
            total_loss_kp += loss_keypoints.item()

            if i % 100 == 0:
                print("Training epoch {}/{}, iteration {}/{}: loss is {}".format(epoch, args.epochs, i, max_iter, loss.item()))

        scheduler.step()
        
        avg_loss = total_loss / iter_count
        avg_keypoint = total_loss_kp / iter_count
        avg_loss_electrode = total_loss_electrode / iter_count
        avg_loss_recon = total_loss_recon / iter_count

        f_train.write(f"{avg_loss}\n")
        f_electrode_train.write(f"{avg_loss_electrode}\n")
        f_recon_train.write(f"{avg_loss_recon}\n")
        f_kp_train.write(f"{avg_keypoint}\n")

        print(f"\033[96mTraining epoch {epoch}/{args.epochs}: avg loss = {avg_loss}\033[0m")

        # validation
        network.eval()
        with torch.no_grad():
            total_loss, total_loss_electrode, total_loss_recon, total_loss_kp, iter_count = 0, 0, 0, 0, 0
            for i, data in enumerate(val_dataloader, 1):
                pc_torso, gt_electrode, mesh_torso_coarse, mesh_torso_dense, mesh_torso_kp = data
                pc_torso, gt_electrode = pc_torso.to(DEVICE), gt_electrode.to(DEVICE) 
                mesh_torso_coarse, mesh_torso_dense = mesh_torso_coarse.to(DEVICE), mesh_torso_dense.to(DEVICE)
                mesh_torso_kp = mesh_torso_kp.to(DEVICE)
                pc_torso = pc_torso.permute(0, 2, 1) 

                out_keypoint, y_coarse, y_detail = network(pc_torso)
                loss_electrode, loss_keypoints = F_loss_electrode(out_keypoint, gt_electrode, mesh_torso_kp)
                loss_recon = F_loss_recon(pc_torso, y_coarse, y_detail, mesh_torso_coarse, mesh_torso_dense)
                loss = loss_electrode + args.lamda*(loss_recon + loss_keypoints)
                
                total_loss += loss.item()
                total_loss_electrode += loss_electrode.item()
                total_loss_recon += loss_recon.item()
                total_loss_kp += loss_keypoints.item()
                iter_count += 1
            
            avg_loss = total_loss / iter_count
            avg_keypoint = total_loss_kp / iter_count
            avg_loss_electrode = total_loss_electrode / iter_count
            avg_loss_recon = total_loss_recon / iter_count

            f_val.write(f"{avg_loss}\n")
            f_electrode_val.write(f"{avg_loss_electrode}\n")
            f_recon_val.write(f"{avg_loss_recon}\n")
            f_kp_val.write(f"{avg_keypoint}\n")
            
            print(f"\033[35mValidation epoch {epoch}/{args.epochs}, loss is {avg_loss}\033[0m")

            if avg_loss < minimum_loss:
                best_epoch = epoch
                minimum_loss = avg_loss
                strNetSaveName = 'net_model.pkl'
                # strNetSaveName = 'net_with_%d.pkl' % epoch
                torch.save(network.state_dict(), os.path.join(args.log_dir, strNetSaveName))

        print("\033[4;37mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))



