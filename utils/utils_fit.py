import os

import torch
import torch.distributed as dist
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np

from utils.utils import get_lr, show_result


def fit_one_epoch(diffusion_model_train, diffusion_model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir, result_dir, test_slices_dict, local_rank=0):
    total_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, imgs in enumerate(gen):
        full_imgs = imgs["fulldose"]
        low_imgs = imgs["lowdose"]
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                full_imgs = full_imgs.cuda(local_rank)
                low_imgs = low_imgs.cuda(local_rank)
        if not fp16:
            optimizer.zero_grad()
            diffusion_loss = torch.mean(diffusion_model_train(x=full_imgs, ax_feature=low_imgs))
            diffusion_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            optimizer.zero_grad()
            with autocast():
                diffusion_loss = torch.mean(diffusion_model_train(x=full_imgs, ax_feature=low_imgs))
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        diffusion_model.update_ema()

        total_loss += diffusion_loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
    
    total_loss = total_loss / epoch_step
    
    if local_rank == 0:
        pbar.close()
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total_loss: %.4f ' % (total_loss))
        loss_history.append_loss(epoch + 1, total_loss = total_loss)
        
        #----------------------------#
        #   每若干个世代保存一次
        #----------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            test_full = test_slices_dict["fulldose"]
            test_low = test_slices_dict["lowdose"]
            with torch.no_grad():
                test_low = test_low.cuda(local_rank)
            print('Show_result:')
            show_result(epoch + 1, diffusion_model, test_low.device, result_dir, test_full, ax_feature=test_low)
            
            torch.save(diffusion_model.state_dict(), os.path.join(save_dir, 'Diffusion_Epoch%d-GLoss%.4f.pth'%(epoch + 1, total_loss)))
            
        torch.save(diffusion_model.state_dict(), os.path.join(save_dir, "diffusion_model_last_epoch_weights.pth"))
