import math
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        if args.kl_anneal_type == "Cyclical":
            self.schedule = self.frange_cycle_linear(
                n_iter=args.num_epoch + 1, 
                start=0.0, 
                stop=1.0, 
                n_cycle=args.kl_anneal_cycle, 
                ratio=args.kl_anneal_ratio
                )
        elif args.kl_anneal_type == "Monotonic":
            # spaciel case for Cyclical annealing with n_cycle = 1
            self.schedule = self.schedule = self.frange_cycle_linear(
                n_iter=args.num_epoch + 1,
                start=0.0,
                stop=1.0,
                n_cycle=1,
                ratio=args.kl_anneal_ratio
                )
        else:
            self.schedule = np.ones(args.num_epoch + 1)

        self.current_epoch = current_epoch
        self.beta = self.schedule[self.current_epoch]
        
    def update(self):
        self.current_epoch += 1
        if self.current_epoch < len(self.schedule):
            self.beta = self.schedule[self.current_epoch]
        else:
            self.beta = self.schedule[-1]
    
    def get_beta(self):
        return max(self.beta, 1e-5)

    # 其他的annealing策略
    # https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        if args.optim == "Adam":
            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        elif args.optim == "AdamW":
            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
        
        if args.fast_train:
            self.scheduler  = optim.lr_scheduler.MultiStepLR(
                self.optim, 
                milestones=[2, 5], 
                gamma=0.1
                )
        elif args.scheduler == "MultiStepLR":
            self.scheduler  = optim.lr_scheduler.MultiStepLR(
                self.optim, 
                milestones=[int(self.args.num_epoch*0.3), int(self.args.num_epoch*0.6)], 
                gamma=0.1
                )
        elif args.scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optim, 
                T_0=int(self.args.num_epoch/6),
                eta_min=1e-5
                )
        elif args.scheduler == "ReduceLROnPlateau":
            self.scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, 
                mode='min', 
                factor=0.65, 
                patience=10, 
                verbose=True
                )
        else:
            raise NotImplementedError(f"Scheduler {args.scheduler} is not implemented")

        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.best_val_loss = float('inf')
        self.log_save_path = os.path.join(
            args.log_save_root, 
            f"lr_{args.lr}_tfr_{args.tfr}_{args.tfr_sde}_{args.tfr_d_step}"\
            f"_kl_{args.kl_anneal_type}_{args.kl_anneal_cycle}_{args.kl_anneal_ratio}"\
            f"_optim_{args.optim}"
            )
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_save_path)
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            train_loss_sum = 0.0
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                train_loss_sum += loss.item()
                beta = self.kl_annealing.get_beta()

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
            train_loss = train_loss_sum / len(train_loader)
            val_loss, all_psnr = self.eval()
            psnr_avg = np.mean(np.concatenate(all_psnr))
            print(f"[Epoch {self.current_epoch}] PSNR: {psnr_avg:.4f}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            self.writer.add_scalar('PSNR', psnr_avg, self.current_epoch)
            self.writer.add_scalar('Train/Loss', train_loss, self.current_epoch)
            self.writer.add_scalar('Val/Loss', val_loss, self.current_epoch)
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], self.current_epoch)
            self.writer.add_scalar('KL_Beta', self.kl_annealing.get_beta(), self.current_epoch)
            self.writer.add_scalar('Teacher_Forcing', self.tfr, self.current_epoch)

            if math.isnan(train_loss):
                print(f"[Warning] NaN loss detected at epoch {self.current_epoch}")
                self.load_checkpoint()
            elif val_loss < self.best_val_loss and not math.isnan(val_loss):
                self.best_val_loss = val_loss
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}_loss_{val_loss}_BEST.ckpt"))
                print(f"Best model updated at epoch {self.current_epoch} with val_loss {val_loss:.4f}")
            elif self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}_loss_{val_loss}.ckpt"))

            self.current_epoch += 1
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        loss_sum = 0.0
        count = 0
        all_psnr = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr_frames = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            all_psnr.append(psnr_frames)
            loss_sum += loss.item()
            count += 1
        
        avg_loss = loss_sum / count

        return avg_loss, all_psnr
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.optim.zero_grad()
        img = img.permute(1, 0, 2, 3, 4) # [K, B, C, H, W]
        label = label.permute(1, 0, 2, 3, 4) # [K, B, C, H, W]
        
        mse_loss = 0
        kl_loss = 0
        last_frame = img[0]

        for t in range(1, self.train_vi_len):
            # Posterior Predictor
            curr_frame = img[t]
            x_prev = self.frame_transformation(last_frame)
            x_in = self.frame_transformation(curr_frame)
            p_in = self.label_transformation(label[t])
            z, mu, logvar = self.Gaussian_Predictor(x_in, p_in)
            
            fusion = self.Decoder_Fusion(x_prev, p_in, z)
            pred_frame = self.Generator(fusion)
            
            mse_loss += self.mse_criterion(pred_frame, curr_frame)
            kl_loss += kl_criterion(mu, logvar, self.batch_size)

            if adapt_TeacherForcing:
                last_frame = curr_frame
            else:
                last_frame = pred_frame.detach()
            
        beta = self.kl_annealing.get_beta()
        loss = (mse_loss + beta * kl_loss) / (self.train_vi_len - 1)
        loss.backward()
        self.optimizer_step()
        
        return loss
    
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4) # [K, B, C, H, W]
        label = label.permute(1, 0, 2, 3, 4) # [K, B, C, H, W]
        
        mse_loss = 0
        last_frame = img[0]
        psnr_frames = []

        for t in range(1, self.val_vi_len):
            gt_frame = img[t]
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            
            x_prev = self.frame_transformation(last_frame)
            p_in = self.label_transformation(label[t])
            
            decoder_input = self.Decoder_Fusion(x_prev, p_in, z)
            pred_frame = self.Generator(decoder_input)
            
            mse_loss += self.mse_criterion(pred_frame, gt_frame)
            psnr_frames.append(Generate_PSNR(pred_frame, gt_frame).item())
            last_frame = pred_frame
            
        return mse_loss / (self.val_vi_len - 1), psnr_frames
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr -= self.tfr_d_step
        self.tfr = max(self.tfr, 0.0)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict":   self.state_dict(),
            "optimizer":    self.optim.state_dict(),  
            "scheduler":    self.scheduler.state_dict(),
            "lr"        :   self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch":   self.current_epoch,
            "kl_annealing": self.kl_annealing.__dict__
        }, path)
        self.args.ckpt_path = path
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        path = self.args.ckpt_path
        if path != None and os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            self.current_epoch = checkpoint['last_epoch']
            
            if 'optimizer' in checkpoint:
                self.optim.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'kl_annealing' in checkpoint:
                self.kl_annealing.__dict__.update(checkpoint['kl_annealing'])

            # self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            # self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            
            print(f"[Reload ckpt] Successfully loaded checkpoint from {path}")

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()
        model.writer.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")

    # Log save root
    parser.add_argument('--log_save_root',      type=str, default="./logs", help="The path to save your log file")

    # Optim & Scheduler setting
    parser.add_argument('--scheduler',           type=str, choices=["MultiStepLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau"], default="MultiStepLR")
    

    

    args = parser.parse_args()
    
    main(args)
