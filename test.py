import sys
import os
import time
from loader import get_loader
import imageio
from micro import TEST
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss
from utils.tools import continue_test,calculate_params_flops
from utils.tools import continue_train, get_logger,set_seed
from train_val_epoch import train_epoch,val_epoch
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.loss_function import get_metrics
from PIL import Image
from matplotlib import pyplot as plt

torch.cuda.set_device(3)
set_seed(42)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    default="BUSI",
    help="input datasets name including ISIC2017, ISIC2018, PH2, Kvasir, or BUSI",
)
parser.add_argument(
    "--imagesize",
    type=int,
    default=256,
    help="input image resolution. 224 for VGG; 256 for Mamba",
)
parser.add_argument(
    "--log",
    type=str,
    default="log",
    help="input log folder: ./log",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default='checkpoints',
    help="the checkpoint path of last model: ./checkpoints",
)
parser.add_argument(
    "--testdir",
    type=str,
    default='Test',
    help="the folder is saving test results",
)


def get_model():
    model=Model(out_channels=[8,16,24,32,40],scale_factor=[1,2,4,8,16])
    model = model.cuda()
    return model



def test_epoch(test_loader,model,criterion,logger,path):
    image_root =  os.path.join(path,'images')
    gt_root =  os.path.join(path,'gt')
    pred_root =  os.path.join(path,'pred')
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(gt_root):
        os.makedirs(gt_root)
    if not os.path.exists(pred_root):
        os.makedirs(pred_root)
    model.eval()
    loss_list=[]
    preds = []
    gts = []
    time_sum=0.0
    id=0.0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, gt,image_name = data
            images, gt = images.cuda().float(), gt.cuda().float()
            time_start = time.time()
            pred = model(images)
            time_end = time.time()
            time_sum = time_sum+(time_end-time_start)
            id=id+1.0
            # 计算损失
            loss = criterion(pred[0],gt)
            #计算损失
            loss_list.append(loss.item())
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy())
    print(time_sum/id)
    log_info,miou=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list),miou



def test(args):
    #init_checkpoint folder
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.datasets)
    #logger
    logger = get_logger('test', os.path.join(os.getcwd(),args.log))
    #initialization cuda
    # set_cuda(gpu_id='6')
    #get loader
    test_loader=get_loader(args.datasets,1,args.imagesize,mode=TEST)
    
    #get model
    model=get_model()
    #calculate parameters and flops
    calculate_params_flops(model,size=args.imagesize,logger=logger)
    #set loss function
    criterion=BceDiceLoss()
    
    #Do continue to run?
    model,_,_=continue_test(model=model,checkpoint_path=checkpoint_path)
    #start to run the model
    test_epoch(test_loader,model,criterion,logger,os.path.join(os.getcwd(),'Test',args.datasets))



if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
