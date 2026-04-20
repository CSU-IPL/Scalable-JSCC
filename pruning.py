import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from utils import args_parser
from main import SemanticComm

args = args_parser()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SemanticComm(args).to(args.device)

args.load = 1

if args.load:
    if not os.path.exists(args.save_path):
        print("No params, start training...")
    else:
        param_dict = torch.load(args.save_path)
        model.load_state_dict(param_dict["gnet_dict"])
        print("Loaded params from {}".format(args.save_path))

total = 0
for m in model.Encoder.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.Encoder.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask1 = []
for k, m in enumerate(model.Encoder.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask1.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total

cfg_enc = cfg


total = 0
for m in model.Decoder1.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.Decoder1.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask2 = []
for k, m in enumerate(model.Decoder1.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask2.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total

cfg_dec1 = cfg


total = 0
for m in model.Decoder2.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.Decoder2.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask2 = []
for k, m in enumerate(model.Decoder2.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask2.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total

cfg_dec2 = cfg

cfg_enc[-1] = 256
cfg_dec1[-1] = 256
cfg_dec2[-1] = 256

print(cfg_enc)
print(cfg_dec1)
print(cfg_dec2)

#
# print('Pre-processing Successful!')
# newmodel = SemanticComm(args, cfg_enc=cfg_enc, cfg_dec1=cfg_dec1, cfg_dec2=cfg_dec2).to(args.device)
#
# layer_id_in_cfg = 0
# start_mask = torch.ones(3)
# end_mask = cfg_mask1[layer_id_in_cfg]
#
# for [m0, m1] in zip(model.Encoder.modules(), newmodel.Encoder.modules()):
#     if isinstance(m0, nn.BatchNorm2d):
#         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#         if idx1.size == 1:
#             idx1 = np.resize(idx1,(1,))
#         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
#         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
#         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
#         m1.running_var = m0.running_var[idx1.tolist()].clone()
#         layer_id_in_cfg += 1
#         start_mask = end_mask.clone()
#         if layer_id_in_cfg < len(cfg_mask1):  # do not change in Final FC
#             end_mask = cfg_mask1[layer_id_in_cfg]
#     elif isinstance(m0, nn.Conv2d):
#         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#         print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#         if idx0.size == 1:
#             idx0 = np.resize(idx0, (1,))
#         if idx1.size == 1:
#             idx1 = np.resize(idx1, (1,))
#         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#         w1 = w1[idx1.tolist(), :, :, :].clone()
#         m1.weight.data = w1.clone()
#     elif isinstance(m0, nn.Linear):
#         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#         if idx0.size == 1:
#             idx0 = np.resize(idx0, (1,))
#         m1.weight.data = m0.weight.data[:, idx0].clone()
#         m1.bias.data = m0.bias.data.clone()
#
#
# layer_id_in_cfg = 0
# start_mask = torch.ones(256)
# end_mask = cfg_mask2[layer_id_in_cfg]
#
# for [m0, m1] in zip(model.Decoder1.modules(), newmodel.Decoder1.modules()):
#     if isinstance(m0, nn.BatchNorm2d):
#         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#         if idx1.size == 1:
#             idx1 = np.resize(idx1,(1,))
#         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
#         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
#         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
#         m1.running_var = m0.running_var[idx1.tolist()].clone()
#         layer_id_in_cfg += 1
#         start_mask = end_mask.clone()
#         if layer_id_in_cfg < len(cfg_mask2):  # do not change in Final FC
#             end_mask = cfg_mask2[layer_id_in_cfg]
#     elif isinstance(m0, nn.Conv2d):
#         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#         print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#         if idx0.size == 1:
#             idx0 = np.resize(idx0, (1,))
#         if idx1.size == 1:
#             idx1 = np.resize(idx1, (1,))
#         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#         w1 = w1[idx1.tolist(), :, :, :].clone()
#         m1.weight.data = w1.clone()
#     elif isinstance(m0, nn.Linear):
#         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#         if idx0.size == 1:
#             idx0 = np.resize(idx0, (1,))
#         m1.weight.data = m0.weight.data[:, idx0].clone()
#         m1.bias.data = m0.bias.data.clone()
#
#
# torch.save({'cfg_enc': cfg_enc,
#             'cfg_dec1': cfg_dec1,
#             'cfg_dec2': cfg_dec2,
#             "epoch": param_dict["epoch"],
#             "lr": param_dict["lr"],
#             "best_psnr": param_dict["best_psnr"],
#             "gnet_dict": newmodel.state_dict()}
#            , os.path.join(args.save, 'pruned_model.pth',))
#
# print(newmodel.Encoder)
# print(newmodel.Decoder1)
# print(newmodel.Decoder2)
# model = newmodel



























