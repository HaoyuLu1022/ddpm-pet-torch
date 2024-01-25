import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.ndimage as ndimage
from numpy.lib.stride_tricks import sliding_window_view

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.callbacks import LossHistory
from utils.dataloader import Diffusion_dataset_collate, DiffusionDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config, preprocess_input
from utils.utils_fit import fit_one_epoch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['NVIDIA_P2P_DISABLE'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5120"

if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #---------------------------------------------------------------------#
    diffusion_model_path    = "model_weights/Diffusion_Flower_mod.pth"
    #---------------------------------------------------------------------#
    #   卷积通道数的设置，显存不够时可以降低，如64
    #---------------------------------------------------------------------#
    channel         = 128
    #---------------------------------------------------------------------#
    #   betas相关参数
    #---------------------------------------------------------------------#
    schedule        = "linear"
    num_timesteps   = 1000
    schedule_low    = 1e-4
    schedule_high   = 0.02
    #---------------------------------------------------------------------#
    #   图像大小的设置，如[128, 128]
    #   设置后在训练时Diffusion的图像看不出来，需要在预测时看单张图像。
    #---------------------------------------------------------------------#
    model_input_shape     = (128, 128)
    img_shape = (256, 200, 200)
    
    #------------------------------#
    #   训练参数设置
    #------------------------------#
    Init_Epoch      = 0
    Epoch           = 500
    batch_size      = 6
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-5
    Min_lr              = 1e-8
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、adamw
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "adamw"
    momentum            = 0.9
    weight_decay        = 0.01
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "poly"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = '/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/ddpm-pet-torch-logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers         = 16
    
    #------------------------------------------#
    #   获得图片路径
    #------------------------------------------#
    annotation_path = "train_slices.txt"

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )
    #------------------------------------------#
    #   Diffusion网络
    #------------------------------------------#
    diffusion_model = GaussianDiffusion(UNet(img_channels=1, condition=True, guide_channels=32, base_channels=channel), model_input_shape, 1, betas=betas)
    # 灰阶图像通道数和预训练模型通道数不一致，通常有两种解决方案
    # 1. 同一(400, 400, 1)切片输入，复制三份形成(400, 400, 3)传入
    # 2. 对模型的第一层各通道参数进行均值操作 <- 

    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if diffusion_model_path != '':
        model_dict      = diffusion_model.state_dict()
        pretrained_dict = torch.load(diffusion_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        diffusion_model.load_state_dict(model_dict)
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    diffusion_model_train = diffusion_model.train()
    
    if Cuda:
        if distributed:
            diffusion_model_train = diffusion_model_train.cuda(local_rank)
            diffusion_model_train = torch.nn.parallel.DistributedDataParallel(diffusion_model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            cudnn.benchmark = True
            diffusion_model_train = torch.nn.DataParallel(diffusion_model)
            diffusion_model_train = diffusion_model_train.cuda()

    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    if local_rank == 0:
        show_config(
            model_input_shape = model_input_shape, Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
            )
    
    #----------------------#
    #   记录Loss
    #----------------------#
    # if local_rank == 0:
    #     time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    #     log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    #     loss_history    = LossHistory(log_dir, [diffusion_model], input_shape=model_input_shape)
    #     result_dir = f"results/train_out_{str(time_str)}"
    #     if not os.path.exists(result_dir):
    #         os.makedirs(result_dir)
    # else:
    #     loss_history    = None
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, [diffusion_model], input_shape=model_input_shape)
    result_dir = f"results/train_out_{str(time_str)}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir) 
    
    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
        }[optimizer_type]
        
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        train_dataset   = DiffusionDataset(lines, model_input_shape, img_shape)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            shuffle         = True
    
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=Diffusion_dataset_collate, sampler=train_sampler, prefetch_factor=2)

        target_shape = (128, 128, 128)
        test_path = 'test_lines.txt'
        with open(test_path) as f:
            lines = f.readlines()
        test_full_dir, test_low_dir = lines[0].split()
        test_full = np.fromfile(test_full_dir, dtype=np.float32)
        test_full, invalid_z_list = preprocess_input(test_full.reshape(img_shape))
        test_full = ndimage.zoom(test_full, [t_shape/img_shape for t_shape, img_shape in zip(target_shape, test_full.shape)])
        test_full_list = np.split(test_full, test_full.shape[0], axis=0)
        test_full_list = [test_full_list[i] for i in range(0, len(test_full_list), len(test_full_list)//4)]

        test_low = np.fromfile(test_low_dir, dtype=np.float32).reshape(img_shape)
        test_low = np.delete(test_low, invalid_z_list, 0)
        test_low /= 5e3
        test_low -= 0.5
        test_low /= 0.5
        test_low = ndimage.zoom(test_low, [t_shape/img_shape for t_shape, img_shape in zip(target_shape, test_low.shape)])
        test_low = np.pad(test_low, ((16, 15), (0, 0), (0, 0)), mode='constant', constant_values=0)
        test_low_list = sliding_window_view(test_low, window_shape=32, axis=0).transpose(0, 3, 1, 2)
        # test_low_list = np.concatenate([np.repeat(np.expand_dims(test_low_list[0, :, :, :], axis=0), 16, axis=0), test_low_list, np.repeat(np.expand_dims(test_low_list[-1, :, :, :], axis=0), 15, axis=0)], axis=0)
        test_low_list = [test_low_list[i, :, :, :] for i in range(len(test_low_list))]
        test_low = torch.stack([torch.Tensor(test_low_list[i].copy()) for i in range(0, len(test_low_list), len(test_low_list)//4)], dim=0)
        test_slices_dict = {
            "fulldose": test_full_list, 
            "lowdose": test_low
        }

        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(diffusion_model_train, diffusion_model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, log_dir, result_dir, test_slices_dict, local_rank)

            if distributed:
                dist.barrier()
