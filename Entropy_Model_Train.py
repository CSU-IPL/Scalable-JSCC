import numpy as np
import os, random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import args_parser, progress_bar
from Networks import HF_Enc, HF_Dec, HFM
from Get_datasets import PairedImageDataset

import torch
from torch import nn
import math
from src.models.image_entropy_models import EntropyBottleneck, GaussianConditional

from src.models.layers import conv3x3, DepthConvBlock2, ResidualBlockUpsample, ResidualBlockWithStride, subpel_conv3x3, MaskedConv2d

# set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class entropy_model(nn.Module):
    def __init__(self,):
        super(entropy_model, self).__init__()
        self.N = 48
        self.h_a = nn.Sequential(
            conv3x3(self.N, self.N),
            nn.LeakyReLU(inplace=True),
            conv3x3(self.N, self.N),
            nn.LeakyReLU(inplace=True),

            conv3x3(self.N, self.N),

        )
        self.h_s = nn.Sequential(
            conv3x3(self.N, self.N),
            nn.LeakyReLU(inplace=True),

            conv3x3(self.N, self.N * 3 // 2),
            nn.LeakyReLU(inplace=True),

            conv3x3(self.N * 3 // 2, self.N * 2),
        )
        self.gaussian_conditional = GaussianConditional(None)

        self.context_prediction = MaskedConv2d(
            self.N, 2 * self.N, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.N * 12 // 3, self.N * 10 // 3, 1),
            # GDN(N * 10 // 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.N * 10 // 3, self.N * 8 // 3, 1),
            # GDN(N * 8 // 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.N * 8 // 3, self.N * 6 // 3, 1),
        )

        self.hyper_cumulative = self.sigmoid_cumulative

    def sigmoid_cumulative(self, x):
        """
        Calculates sigmoid of the tensor to use as a replacement of CDF
        """
        return torch.sigmoid(x)

    def hyperlatent_rate(self, z):
        """
        Calculate hyperlatent rate

        Since we assume that each latent is modelled a Non-parametric convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)

        See apeendix 6.2
        J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
        “Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
        Available: https://openreview.net/forum?id=rkcQFMZRb.
        """
        upper = self.hyper_cumulative(z + .5)
        lower = self.hyper_cumulative(z - .5)
        return -torch.sum(torch.log2(torch.abs(upper - lower)), dim=(1, 2, 3))

    def feature_probs_based_sigma(self, feature, mean, sigma):
        outputs = self.quantize(
            feature, "dequantize", mean
        )
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.mean(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        tensor_bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)
        return total_bits, tensor_bits,  probs

    def quantize(self, inputs, mode, means=None):
        assert (mode == "dequantize")
        outputs = inputs.clone()
        outputs -= means
        outputs = torch.round(outputs)
        outputs += means
        return outputs

    def forward(self, y):
        z = self.h_a(y)
        hyperlatent_rate = torch.mean(self.hyperlatent_rate(z))
        params = self.h_s(z)
        y_tail = self.gaussian_conditional._quantize(y, "noise")
        y_tail = y_tail.float()
        ctx_params = self.context_prediction(y_tail)
        params = params.float()
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        gaussian_params = gaussian_params.float()
        scales_tail, means_tail = gaussian_params.chunk(2, 1)
        entropy, tensor_rate, _ = self.feature_probs_based_sigma(y_tail, means_tail, scales_tail)

        return (entropy + hyperlatent_rate), tensor_rate


class SemanticComm(nn.Module):
    def __init__(self, args):
        super(SemanticComm, self).__init__()
        self.N = 48
        self.args = args
        self.HFM = HFM()
        self.HF_Enc = HF_Enc(3, 48)
        self.HF_Dec = HF_Dec(48, 3)
        self.entropy_model = entropy_model()

    def power_norm(self, feature, target_power=1.0):
        in_shape = feature.shape
        sig_in = feature.view(in_shape[0], -1)

        power = torch.mean(sig_in ** 2, dim=1, keepdim=True)

        sig_out = sig_in / torch.sqrt(power + 1e-8) * torch.sqrt(torch.tensor(target_power, device=feature.device))

        return sig_out.view(in_shape)

    def forward(self, y):
        y = self.HFM(y)
        ori_high = y
        y = self.HF_Enc(y)
        y = self.power_norm(y)

        rate, tensor_rate = self.entropy_model(y)

        rec_high = self.HF_Dec(y)
        return ori_high, rec_high, rate, tensor_rate


def train(epoch, args, model, trainloader, best_PSNR):
    # ============================================= training
    print('\nEpoch: %d' % epoch)
    model.train()
    total_loss = 0
    for batch_idx, (LR_image) in enumerate(trainloader):

        inputs = LR_image.to(args.device)

        args.optimizer.zero_grad()
        ori_high, rec_high, rate, tensor_rate = model(inputs)

        loss = 0.025 * args.loss(ori_high, rec_high) + rate

        total_loss += loss
        loss.backward()
        args.optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'MSE: %.4f' % (loss))


def tes(epoch, args, model, testloader, best_PSNR, saveflag = 1):
    model.eval()
    psnr_all_list = []
    MSEnoAvg = nn.MSELoss(reduction = 'none')
    with torch.no_grad():
        for batch_idx, (LR_image) in enumerate(testloader):
            inputs = LR_image.to(args.device)
            b, c, h, w = inputs.shape[0],inputs.shape[1],inputs.shape[2], inputs.shape[3]
            inputs = inputs.to(args.device)

            ori_high, rec_high, rate, tensor_rate = model(inputs)

            loss = MSEnoAvg(ori_high, rec_high)

            MSE_each_image = (torch.sum(loss.reshape(b, c*h*w),dim=1))/(c*h*w)

            one_batch_PSNR = MSE_each_image.data.cpu().numpy()
            psnr_all_list.extend(one_batch_PSNR)

        test_PSNR=np.mean(psnr_all_list)
        test_PSNR=np.around(test_PSNR,5)
        
        print("MSE=", test_PSNR)
        # print(str(test_PSNR) + ",", end="")

    if saveflag == 1:
        # Save checkpoint.
        if test_PSNR < best_PSNR:
            print('Saving..')
            state = {
                'HF_Enc': model.HF_Enc.state_dict(),
                'HF_Dec': model.HF_Dec.state_dict(),
                'Entropy_Model': model.entropy_model.state_dict()
            }

            filename = "Entropy_model_128"

            if not os.path.isdir('models/entropy model(no channel)'):
                os.mkdir('models/entropy model(no channel)')
            torch.save(state, './entropy model(no channel)/' + filename + '.pth')
            best_PSNR = test_PSNR

        return best_PSNR
    return test_PSNR

from Get_datasets import OneImageDataset

def main(model, args):
    # ======================================================= load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dir = "./datasets/CelebA/train/128"
    test_dir = "./datasets/CelebA/val/128"

    train_dataset = OneImageDataset(train_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = OneImageDataset(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # ======================================================= start (train or val)
    if args.load == 0:
        best_PSNR = 1
        for epoch in np.arange(args.numepoch):
            train(epoch, args, model, train_loader, best_PSNR)
            best_PSNR = tes(epoch, args, model, test_loader, best_PSNR)

            args.scheduler.step()
    else:
        filename = "./Super_Resolution/" + "snr10_rate0.12.pth"
        checkpoint = torch.load(filename)

        model.load_state_dict(checkpoint['model'])

        tes(0, args, model, test_loader, 0, saveflag=0)

if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()

    args.snr = 10
    args.load = 0

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.loss = nn.MSELoss()
    # ======================================================= Initialize the model
    model = SemanticComm(args).to(args.device)
    if args.device == 'cuda:0':
        if args.load == 0:
            print('gpu')
        torch.backends.cudnn.benchmark = True

    # ======================================================= Optimizer
    if args.adamW == 1:
        args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
    else:
        args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # ======================================================= lr scheduling
    # lambdafn = lambda epoch: (1-epoch/args.numepoch)
    lambdafn = lambda epoch: 1.0
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambdafn)
    main(model, args)


