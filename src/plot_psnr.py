import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Trainer import VAE_Model
from dataloader import Dataset_Dance

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',    type=int,    default=1)
    parser.add_argument('--lr',            type=float,  default=0.001)
    parser.add_argument('--device',        type=str,    default="cuda")
    parser.add_argument('--optim',         type=str,    default="Adam")
    parser.add_argument('--DR',            type=str,    required=True,  help="Dataset Path")
    parser.add_argument('--ckpt_path',     type=str,    required=True,  help="Checkpoint Path")
    parser.add_argument('--plot_save_root',type=str,    default="psnr_plots")
    parser.add_argument('--frame_H',       type=int,    default=32)
    parser.add_argument('--frame_W',       type=int,    default=64)
    parser.add_argument('--train_vi_len',  type=int,    default=16)
    parser.add_argument('--val_vi_len',    type=int,    default=630)
    
    # Module parameters (should match training)
    parser.add_argument('--F_dim',         type=int, default=128)
    parser.add_argument('--L_dim',         type=int, default=32)
    parser.add_argument('--N_dim',         type=int, default=12)
    parser.add_argument('--D_out_dim',     type=int, default=192)
    
    # Dummy args for VAE_Model initialization
    parser.add_argument('--kl_anneal_type',  type=str, default='Cyclical')
    parser.add_argument('--kl_anneal_cycle', type=int, default=10)
    parser.add_argument('--kl_anneal_ratio', type=float, default=1)
    parser.add_argument('--num_epoch',       type=int, default=70)
    parser.add_argument('--tfr',             type=float, default=1.0)
    parser.add_argument('--tfr_sde',         type=int,   default=10)
    parser.add_argument('--tfr_d_step',      type=float, default=0.1)
    parser.add_argument('--fast_train',      action='store_true')
    parser.add_argument('--scheduler',       type=str, default="MultiStepLR")
    parser.add_argument('--log_save_root',   type=str, default="./logs")
    parser.add_argument('--save_root',       type=str, default="./checkpoints")
    parser.add_argument('--num_workers',     type=int, default=4)
    parser.add_argument('--fast_train_epoch',type=int, default=5)

    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    # Load Model
    model = VAE_Model(args).to(device)
    model.load_checkpoint()
    model.eval()

    # Load Data
    transform = transforms.Compose([
        transforms.Resize((args.frame_H, args.frame_W)),
        transforms.ToTensor()
    ])
    dataset = Dataset_Dance(root=args.DR, transform=transform, mode='val', video_len=args.val_vi_len, partial=1.0)  
    val_loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    all_psnr_sequences = []

    print("Starting evaluation...")
    with torch.no_grad():
        for (img, label) in tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            # Use the same logic as Trainer.val_one_step
            _, psnr_frames = model.val_one_step(img, label)
            all_psnr_sequences.append(psnr_frames)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # If there are multiple sequences in validation set, we can plot them or average them.
    # Usually for PSNR-per-frame diagram, we plot the average across sequences if multiple exist,
    # or just one if it's the standard validation sequence.
    
    avg_psnr_per_frame = np.mean(all_psnr_sequences, axis=0)
    frames = np.arange(1, len(avg_psnr_per_frame) + 1)

    plt.plot(frames, avg_psnr_per_frame, label='Average PSNR')
    plt.xlabel('Frame index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR-per-frame diagram in the validation dataset')
    plt.grid(True)
    plt.legend()
    
    os.makedirs(args.plot_save_root, exist_ok=True)
    save_path = os.path.join(args.plot_save_root, 'psnr_per_frame_plot.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
