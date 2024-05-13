## DDPM-PET-torch：使用2维DDPM对3维低剂量PET图像降噪
---
> Forked from https://github.com/bubbliiiing/ddpm-pytorch（DDPM: Denoising Diffusion Probabilistic Models模型在pytorch当中的实现）. 

### 目录
1. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
参考`env.yaml`文件。pytorch尽量选择2.0以上版本

## 文件下载
为了验证模型的有效性，原repo使用了**花的例子**进行了预训练。训练好的生成器模型[Diffusion_Flower.pth](https://github.com/bubbliiiing/ddpm-pytorch/releases/download/v1.0/Diffusion_Flower.pth)可以通过百度网盘下载或者通过GITHUB下载    
权值的百度网盘地址如下：    
链接: https://pan.baidu.com/s/1AI5jB0OPYbLGAX4JLbotXA 提取码: kbtp     

花的数据集可以通过百度网盘下载：   
链接: https://pan.baidu.com/s/1ITA1Lw_K28B3nbNPnI3_Kw 提取码: 11yt  

## 预测步骤
1. 按照训练步骤训练。    
2. 在`ddpm.py`文件里，在如下部分修改model_path使其对应训练好的文件；也可以在`predict.py`里，创建`diffusion`对象时传入指定`model_path`。**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    #-----------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #-----------------------------------------------#
    "model_path"        : 'model_data/Diffusion_Flower.pth',
    #-----------------------------------------------#
    #   卷积通道数的设置
    #-----------------------------------------------#
    "channel"           : 128,
    #-----------------------------------------------#
    #   输入图像大小的设置
    #-----------------------------------------------#
    "input_shape"       : (64, 64),
    #---------------------------------------------------------------------#
    #   betas相关参数
    #---------------------------------------------------------------------#
    "schedule"          : "linear",
    "num_timesteps"     : 1000,
    "schedule_low"      : 1e-4,
    "schedule_high"     : 0.02,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
```python
ddpm = Diffusion(model_path=model_path, guide_channels=ax_channel_num, loss_type="l2")
```

3. 运行`predict.py`，终端会询问`model_path`、生成模式（DDPM、DDIM或DPM-Solver）、采样步数（针对DDIM和DPM-Solver）；输入后即可生成对应3维图像[^1]

   [^1]: 本项目开发时为了方便，默认对`test_lines.txt`中第一个路径对应的低剂量图像进行降噪。若有需要可以自行实现指定低剂量图像路径功能；或使用`ddpm.py`中的`show_result_3d_loop`函数，遍历所有`test_lines.txt`中的低剂量图像。

   ，存储在`results/predict_out`路径下。

## 训练步骤
1. 准备好lowdose和fulldose的3维图像数据集  
2. 在`txt_annotation.py`中填写数据集路径并运行，生成`train_lines.txt`，确保`train_lines.txt`内部是有文件路径内容的。  
3. 在`preprocess_slice.py`中修改作为条件的邻近切片数量（默认为32），指定切片存储路径，后运行即可得到切片后的数据集
4. 在`slice_annotation.py`中修改切片数据集路径并运行，生成`train_slices.txt`，确保`train_slices.txt`内部是有文件路径内容的。  
5. 运行`train.py`文件进行训练，训练过程中生成的图片可查看`results/loss_<对应训练时间>/train_out`文件夹下的图片（推荐只用于观察是否能生成正确图片，而不用于模型质量评估；质量评估尽量使用上述`predict.py`测试流程）。  
