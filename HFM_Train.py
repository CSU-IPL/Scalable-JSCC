import numpy as np
import os, random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import args_parser, progress_bar
from Networks import HF_Enc, HF_Dec, HFM
import torch
from torch import nn
from Get_datasets import PairedImageDataset, OneImageDataset
from Entropy_Model_Train import entropy_model

# set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class SemanticComm(nn.Module):
    def __init__(self, args):
        super(SemanticComm, self).__init__()
        self.N = int(args.rate * 48)
        self.args = args
        self.HFM = HFM()
        self.HF_Enc = HF_Enc(3, 48)
        self.HF_Dec = HF_Dec(48, 3)

        self.entropy_model = entropy_model()
        self.entropy_model_PATH = "models/entropy model(no channel)/Entropy_model_128.pth"

        self.load_Entropy_Model(self.entropy_model_PATH)

    def load_Entropy_Model(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        self.entropy_model.load_state_dict(checkpoint['Entropy_Model'])
        for param in self.entropy_model.parameters():
            param.requires_grad = False

    def generate_mask(self, tensor_rate, percent):

        channel_sums = tensor_rate.view(tensor_rate.size(0), tensor_rate.size(1), -1).sum(dim=-1)

        k = int(channel_sums.size(1) * percent)

        _, topk_indices = torch.topk(channel_sums, k, dim=1)

        mask = torch.zeros_like(channel_sums, dtype=torch.int32).scatter_(1, topk_indices, 1)

        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(tensor_rate)
        return mask

    def power_norm(self, feature, target_power=1.0):
        in_shape = feature.shape
        sig_in = feature.view(in_shape[0], -1)

        power = torch.mean(sig_in ** 2, dim=1, keepdim=True)

        sig_out = sig_in / torch.sqrt(power + 1e-8) * torch.sqrt(torch.tensor(target_power, device=feature.device))

        return sig_out.view(in_shape)

    def channel(self, data_x):
        inputBS, len_data_x = data_x.size(0), data_x.size(1)

        noise_std = 10 ** (-self.args.snr * 1.0 / 10 / 2)

        # real channel
        AWGN = torch.normal(0, std=noise_std, size=data_x.size()).to(self.args.device)

        if self.args.channel_type == "awgn":
            return data_x + AWGN
        elif self.args.channel_type == "rayleigh":
            self.hh = self.hh.to(self.args.device)
            return self.hh * data_x + AWGN
        else:
            return False

    def forward(self, y):
        y = self.HFM(y)
        ori_high = y
        y = self.HF_Enc(y)

        _, tensor_rate = self.entropy_model(y)
        random_number = 0.5
        if self.args.load == 0:
            self.args.snr = float(int(random.choice([3, 5, 7, 9, 11, 13])))
            random_number = random.choice([0 / 48, 3 / 48, 6 / 48])
            self.hh = torch.sqrt(torch.normal(mean=0.0, std=1, size=[1]) ** 2
                                        + torch.normal(mean=0.0, std=1, size=[1]) ** 2) / np.sqrt(2)

        y = self.power_norm(y)

        mask = self.generate_mask(tensor_rate, random_number)
        y = y * mask

        y = self.channel(y)

        rec_high = self.HF_Dec(y)
        return ori_high, rec_high


def train(epoch, args, model, trainloader, best_PSNR):
    # ============================================= training
    print('\nEpoch: %d' % epoch)
    model.train()
    total_loss = 0
    for batch_idx, (LR_image) in enumerate(trainloader):

        inputs = LR_image.to(args.device)

        args.optimizer.zero_grad()
        ori_high, rec_high = model(inputs)

        loss = args.loss(ori_high, rec_high)

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

            ori_high, rec_high = model(inputs)

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
                'HF_Dec': model.HF_Dec.state_dict()
            }

            filename = "random_HFM128"

            if not os.path.isdir('models/awgn'):
                os.mkdir('models/awgn')
            torch.save(state, './awgn/' + filename + '.pth')
            best_PSNR = test_PSNR

        return best_PSNR
    return test_PSNR

def main(model, args):
    # ======================================================= load datasets
    transform = transforms.Compose([
        # transforms.Resize((1024, 1024)),
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
    args.rate = 1.0
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


