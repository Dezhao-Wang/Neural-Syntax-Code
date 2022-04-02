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

class DIV2KDataset(torch.utils.data.Dataset):
  def __init__(self, train_glob, transform):
    super(DIV2KDataset, self).__init__()
    self.transform = transform
    self.images = list(sorted(glob.glob(train_glob)))
    
  def __getitem__(self, idx):
    img_path = self.images[idx]
    img = Image.open(img_path).convert("RGB")
    img = self.transform(img)
    return img
  
  def __len__(self):
    return len(self.images)

class Preprocess(object):
  def __init__(self):
    pass

  def __call__(self, PIL_img):
    img = np.asarray(PIL_img, dtype=np.float32)
    img /= 127.5
    img -= 1.0
    return img.transpose((2, 0, 1))
    
def quantize_image(img):
  img = torch.clamp(img, -1, 1)
  img += 1
  img = torch.round(img)
  img = img.to(torch.uint8)
  return img



def train(train_data_path, lmbda, lr, batch_size, checkpoint_dir, weight_path, is_high, post_processing):

  train_data = DIV2KDataset(train_data_path,transform=tv.transforms.Compose([
      tv.transforms.RandomCrop(256),
      Preprocess()
    ]))
  
  training_loader = torch.utils.data.DataLoader(train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=8)

  net = Net((batch_size,256,256,3), (1,256,256,3), is_high, post_processing).cuda()

  def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)  
  
  if weight_path != "":
    net.load_state_dict(torch.load(weight_path),strict=True)
  else:
    net.apply(weight_init)
  
  
  if post_processing: # Only Train Post Processing Module
    opt = optim.Adam(net.post_processing_params(), lr=lr)
    sch = optim.lr_scheduler.MultiStepLR(opt, [1200, 1350], 0.5)
    train_epoch = 1500
  else:
    opt = optim.Adam(net.base_params(), lr=lr)
    sch = optim.lr_scheduler.MultiStepLR(opt, [4000, 4500, 4750], 0.5)
    train_epoch = 5000

  net = nn.DataParallel(net)
  
  for epoch in range(0, train_epoch):
    net.train()
    
    start_time = time.time()

    list_train_loss = 0.
    list_train_bpp = 0.
    list_train_mse = 0.

    cnt = 0

    for i, data in enumerate(training_loader, 0):

      x = data.cuda()
      opt.zero_grad()
      train_bpp, train_mse = net(x, 'train')

      train_loss = lmbda * train_mse + train_bpp
      train_loss = train_loss.mean()
      train_bpp = train_bpp.mean()
      train_mse = train_mse.mean()

      if np.isnan(train_loss.item()):
        raise Exception('NaN in loss')

      list_train_loss += train_loss.item()
      list_train_bpp += train_bpp.item()
      list_train_mse += train_mse.item()

      train_loss.backward()
      nn.utils.clip_grad_norm_(net.parameters(), 10)
      opt.step()
      cnt += 1

    print('[Epoch %04d TRAIN] Loss: %.4f bpp: %.4f mse: %.4f  ' % (
      epoch,
      list_train_loss / cnt,
      list_train_bpp / cnt,
      list_train_mse / cnt
      )
    )
    
    sch.step()

    
    if (epoch % 100 == 99):

      print('[INFO] Saving')
      if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
      torch.save(net.module.state_dict(), '%s/%04d.ckpt' % (checkpoint_dir, epoch))
    
    with open(os.path.join(checkpoint_dir, 'train_log.txt'), 'a') as fd:
      fd.write('[Epoch %04d TRAIN] Loss: %.4f bpp: %.4f mse: %.4f \n' % (epoch,list_train_loss / cnt,list_train_bpp / cnt,list_train_mse / cnt))
    fd.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "--train_data_path", default="../DIV2K",
      help="Directory of Testset Images")
  parser.add_argument(
      "--weight_path", default="",
      help="Path of Pretrained Checkpoint")
  parser.add_argument(
      "--checkpoint_dir", default="./ckpt",
      help="Directory of Saved Checkpoints")
  parser.add_argument(
      "--high",  action="store_true",
      help="Using High Bitrate Model")
  parser.add_argument(
      "--post_processing", action="store_true",
      help="Using Post Processing")
  parser.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--lr", type=float, default=1e-4,
      help="Learning Rate")
  parser.add_argument(
      "--batch_size", type=float, default=8,
      help="Batch Size")
  
  args = parser.parse_args()

  train(args.train_data_path, args.lmbda, args.lr, args.batch_size, args.checkpoint_dir, args.weight_path, args.high, args.post_processing)