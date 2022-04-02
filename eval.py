import argparse
import glob

import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F

import pickle
from PIL import Image
from torch.autograd import Function
import time
import os

from model.net import Net 


def val(data_path, weight_path, lmbda, is_high, post_processing, pre_processing, tune_iter):

  if pre_processing is False:
    
    list_eval_bpp = 0.
    list_v_psnr = 0.
    cnt = 0
    sum_time = 0.
    for img_name in os.listdir(data_path):
      data = Image.open(os.path.join(data_path,img_name))
      data = tv.transforms.ToTensor()(data)
      
      _,h,w = data.shape 
      h_padded = h
      w_padded = w
      if h % 64 != 0:
        h_padded = (h//64) * 64 + 64
      if w % 64 != 0:
        w_padded = (w//64) * 64 + 64
      padding_h = h_padded - h
      padding_w = w_padded - w

      h_pad_zeros = torch.ones(3,padding_h,w)
      w_pad_zeros = torch.ones(3,h_padded, padding_w )
      data = torch.cat((data,h_pad_zeros), 1)
      data = torch.cat((data,w_pad_zeros), 2)        

      data = data.unsqueeze(0)
      data = data*2.0 - 1.0

      data_bpp = 0.0
      data_mse = 0.0
      
      net = Net((1,h,w,3), (1,h,w,3), is_high, post_processing).cuda()
      net.load_state_dict(torch.load(weight_path),strict=True)
      
      begin_time = time.time()

      with torch.no_grad():
        eval_bpp,v_mse, v_psnr = net(data.cuda(), 'test')
      
      end_time = time.time()

      sum_time += end_time-begin_time
      
      list_eval_bpp += eval_bpp.mean().item()
      list_v_psnr += v_psnr.mean().item()
      
      print(end_time-begin_time,img_name,eval_bpp.mean().item(),v_psnr.mean().item(), (eval_bpp + lmbda * v_mse).cpu().item() )

      cnt += 1
    
    print('[WITHOUT PRE-PROCESSING] ave_time:%.4f bpp: %.4f psnr: %.4f' % (
      sum_time / cnt,
      list_eval_bpp / cnt, 
      list_v_psnr / cnt
      )
    )
    
  else: # Pre Processing is True
    list_eval_bpp = 0.
    list_v_psnr = 0.
    cnt = 0
    for img_name in os.listdir(data_path):
      begin_time = time.time()

      data = Image.open(os.path.join(data_path,img_name))
      data = tv.transforms.ToTensor()(data)

      _,h,w = data.shape 
      h_padded = h
      w_padded = w
      if h % 64 != 0:
        h_padded = (h//64) * 64 + 64
      if w % 64 != 0:
        w_padded = (w//64) * 64 + 64
      padding_h = h_padded - h
      padding_w = w_padded - w

      h_pad_zeros = torch.ones(3,padding_h,w)
      w_pad_zeros = torch.ones(3,h_padded, padding_w )
      data = torch.cat((data,h_pad_zeros), 1)
      data = torch.cat((data,w_pad_zeros), 2)   

      data = data.unsqueeze(0)
      data = data*2 - 1

      data_bpp = 0.0
      data_mse = 0.0

      net = Net((1,h,w,3), (1,h,w,3), is_high, post_processing).cuda()
      net.load_state_dict(torch.load(weight_path),strict=True)
      opt_enc = optim.Adam(net.a_model.parameters(), lr=1e-5)
      sch = optim.lr_scheduler.MultiStepLR(opt_enc,[50], 0.5)
      
      net.post_processing = False # Update encoder without post-processing to save GPU memory
                                  # If the GPU memory is sufficient, you can delete this sentence
      for iters in range(tune_iter):
        train_bpp, train_mse = net(data.cuda(), 'train')

        train_loss = lmbda * train_mse + train_bpp
        train_loss = train_loss.mean()

        opt_enc.zero_grad()
        train_loss.backward()
        opt_enc.step()

        sch.step()

      net.post_processing = post_processing
      with torch.no_grad():
        eval_bpp,_,v_psnr = net(data.cuda(), 'test')

      list_eval_bpp += eval_bpp.mean().item()
      list_v_psnr += v_psnr.mean().item()
      cnt += 1
      
      print([time.time()-begin_time], img_name,eval_bpp.mean().item(),v_psnr.mean().item())

    print('[WITH PRE-PROCESSING] bpp: %.4f psnr: %.4f' % (
      list_eval_bpp / cnt, 
      list_v_psnr / cnt
      )
    )





if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument(
      "--data_path", default="../Kodak",
      help="Directory of Testset Images")
  parser.add_argument(
      "--weight_path", default="./test.ckpt",
      help="Path of Checkpoint")
  parser.add_argument(
      "--high",  action="store_true",
      help="Using High Bitrate Model")
  parser.add_argument(
      "--post_processing", action="store_true",
      help="Using Post Processing")
  parser.add_argument(
      "--pre_processing", action="store_true",
      help="Using Pre Processing (Online Finetuning)")
  parser.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--tune_iter", type=int, default=100,
      help="Finetune Iteration")
  
  args = parser.parse_args()

  val(args.data_path, args.weight_path, args.lmbda, args.high, args.post_processing, args.pre_processing, args.tune_iter)

