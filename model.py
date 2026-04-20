import os, random

from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from channel import *
from matplotlib import pyplot as plt
import torch
from Networks import (Adapter1, Adapter2, Adapter3, Encoder, Hy_Enc1, Hy_Enc2, Hy_Enc3, Hy_Dec1, Hy_Dec2, Hy_Dec3, HFM, HF_Enc, HF_Dec,
                      Decoder1, Decoder2, Decoder3, Rev1, Rev2, Rev3, Discriminator, HF_Ref1, HF_Ref2, HF_Ref3, Fusion)
from utils import *
from Entropy_Model_Train import entropy_model
import torch.nn.functional as F


def one_image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def get_cfg_by_ratio(ratio):
    cfg_dict = {
        0.5: {
            "cfg_enc":  [119, 148, 164, 142, 149, 132, 132, 142, 146, 126, 256],
            "cfg_dec1": [125, 125, 140, 256],
            "cfg_dec2": [111, 136, 137, 256],
            "cfg_dec3": [130, 143, 123, 256],
        },
        0.7: {
            "cfg_enc":  [75, 63, 52, 70, 79, 78, 80, 84, 111, 102, 256],
            "cfg_dec1": [58, 79, 88, 256],
            "cfg_dec2": [49, 61, 83, 256],
            "cfg_dec3": [99, 99, 54, 256],
        },
        0.8: {
            "cfg_enc": [75, 63, 52, 70, 79, 78, 80, 84, 111, 102, 256],
            "cfg_dec1": [58, 79, 88, 256],
            "cfg_dec2": [49, 61, 83, 256],
            "cfg_dec3": [99, 99, 54, 256],
        },
        1.0: {
            "cfg_enc":  None,
            "cfg_dec1": None,
            "cfg_dec2": None,
            "cfg_dec3": None,
        },
    }

    if ratio not in cfg_dict:
        raise ValueError(f"Unsupported ratio {ratio}. Available: {list(cfg_dict.keys())}")

    return cfg_dict[ratio]

class SemanticComm(nn.Module):
    def __init__(self, args, cfg_dict):
        super(SemanticComm, self).__init__()
        self.args = args

        self.Fusion = Fusion()

        self.Adapter1 = Adapter1()
        self.Adapter2 = Adapter2()
        self.Adapter3 = Adapter3()

        cfg_enc = cfg_dict["cfg_enc"]
        cfg_dec1 = cfg_dict["cfg_dec1"]
        cfg_dec2 = cfg_dict["cfg_dec2"]
        cfg_dec3 = cfg_dict["cfg_dec3"]
        self.Encoder = Encoder(cfg_enc)
        self.Decoder1 = Decoder1(cfg_dec1)
        self.Decoder2 = Decoder2(cfg_dec2)
        self.Decoder3 = Decoder3(cfg_dec3)

        self.Hy_Enc1 = Hy_Enc1(256, 5)
        self.Hy_Dec1 = Hy_Dec1(5, 256)

        self.Hy_Enc2 = Hy_Enc2(256, 4)
        self.Hy_Dec2 = Hy_Dec2(4, 256)

        self.Hy_Enc3 = Hy_Enc3(256, 8)
        self.Hy_Dec3 = Hy_Dec3(8, 256)

        self.HFM = HFM()

        self.HF_Ref1 = HF_Ref1(3, 128)
        self.HF_Ref2 = HF_Ref2(3, 128)
        self.HF_Ref3 = HF_Ref3(3, 128)

        self.HF_Enc1 = HF_Enc(3, 48)
        self.HF_Dec1 = HF_Dec(48, 3)

        self.HF_Enc2 = HF_Enc(3, 48)
        self.HF_Dec2 = HF_Dec(48, 3)

        self.HF_Enc3 = HF_Enc(3, 48)
        self.HF_Dec3 = HF_Dec(48, 3)

        self.entropy_model1 = entropy_model()
        self.entropy_model_PATH1 = "./models/entropy model(no channel)/Entropy_model_64.pth"
        self.entropy_model2 = entropy_model()
        self.entropy_model_PATH2 = "./models/entropy model(no channel)/Entropy_model_128.pth"
        self.entropy_model3 = entropy_model()
        self.entropy_model_PATH3 = "./models/entropy model(no channel)/Entropy_model_32.pth"

        if args.channel_type == 'awgn':
            self.HFM_PATH1 = "./models/awgn/random_HFM64awgn.pth"
            self.HFM_PATH2 = "./models/awgn/random_HFM128awgn.pth"
            self.HFM_PATH3 = "./models/awgn/random_HFM32awgn.pth"
        elif args.channel_type == 'rayleigh':
            self.HFM_PATH1 = "./models/slow rayleigh/random_HFM64rayleigh.pth"
            self.HFM_PATH2 = "./models/slow rayleigh/random_HFM128rayleigh.pth"
            self.HFM_PATH3 = "./models/slow rayleigh/random_HFM32rayleigh.pth"
        else:
            print("Unsupported channel type")
            exit(1)
        self.load_Entropy_Model(self.entropy_model1, self.entropy_model_PATH1)
        self.load_HFM_Enc(self.HF_Enc1, self.HFM_PATH1)
        self.load_HFM_Dec(self.HF_Dec1, self.HFM_PATH1)

        self.load_Entropy_Model(self.entropy_model2, self.entropy_model_PATH2)
        self.load_HFM_Enc(self.HF_Enc2, self.HFM_PATH2)
        self.load_HFM_Dec(self.HF_Dec2, self.HFM_PATH2)

        self.load_Entropy_Model(self.entropy_model3, self.entropy_model_PATH3)
        self.load_HFM_Enc(self.HF_Enc3, self.HFM_PATH3)
        self.load_HFM_Dec(self.HF_Dec3, self.HFM_PATH3)

        self.Rev1 = Rev1(256)
        self.Rev2 = Rev2(256)
        self.Rev3 = Rev3(256)

        self.channel = Channel(args)

    def forward(self, Inputs):
        if self.args.is_training:
            self.args.csnr = float(int(random.choice([3, 5, 7, 9, 11, 13])))
            self.args.random_number = random.choice([0 / 48, 3 / 48, 6 / 48])
            self.channel.h = torch.sqrt(torch.normal(mean=0.0, std=1, size=[1]) ** 2
                                        + torch.normal(mean=0.0, std=1, size=[1]) ** 2) / np.sqrt(2)
        # self.channel.h = torch.sqrt(torch.normal(mean=0.0, std=1, size=[1]) ** 2
        #                             + torch.normal(mean=0.0, std=1, size=[1]) ** 2) / np.sqrt(2)
        img128 = Inputs
        img64 = self.Generate_LR(img128, upscale_factor=2)
        img32 = self.Generate_LR(img128, upscale_factor=4)

        ori_high_64 = self.HFM(img64)
        input_high_64 = self.HF_Ref1(ori_high_64)
        compress_high_64 = self.HF_Enc1(ori_high_64)

        ori_high_128 = self.HFM(img128)
        input_high_128 = self.HF_Ref2(ori_high_128)
        compress_high_128 = self.HF_Enc2(ori_high_128)

        ori_high_32 = self.HFM(img32)
        input_high_32 = self.HF_Ref3(ori_high_32)
        compress_high_32 = self.HF_Enc3(ori_high_32)

        img64 = self.Adapter1(img64, input_high_64)
        img128 = self.Adapter2(img128, input_high_128)
        img32 = self.Adapter3(img32, input_high_32)

        concat = self.Fusion(img64, img128, img32)
        concat = self.Encoder(concat)

        img64 = self.Hy_Enc1(concat)
        img128 = self.Hy_Enc2(concat)
        img32 = self.Hy_Enc3(concat)

        _, tensor_rate_64 = self.entropy_model1(compress_high_64)
        _, tensor_rate_128 = self.entropy_model2(compress_high_128)
        _, tensor_rate_32 = self.entropy_model3(compress_high_32)

        mask_64 = self.generate_mask(tensor_rate_64, self.args.random_number)
        mask_128 = self.generate_mask(tensor_rate_128, self.args.random_number)
        mask_32 = self.generate_mask(tensor_rate_32, self.args.random_number)

        compress_high_hat_64 = self.channel(compress_high_64, self.args.csnr)
        compress_high_hat_128 = self.channel(compress_high_128, self.args.csnr)
        compress_high_hat_32 = self.channel(compress_high_32, self.args.csnr)

        compress_high_hat_64 = compress_high_hat_64 * mask_64
        compress_high_hat_128 = compress_high_hat_128 * mask_128
        compress_high_hat_32 = compress_high_hat_32 * mask_32

        img64 = self.channel(img64, self.args.csnr)
        img128 = self.channel(img128, self.args.csnr)
        img32 = self.channel(img32, self.args.csnr)

        rec_high_64 = self.HF_Dec1(compress_high_hat_64)
        input_high_hat_64 = self.HF_Ref1(rec_high_64)

        rec_high_128 = self.HF_Dec2(compress_high_hat_128)
        input_high_hat_128 = self.HF_Ref2(rec_high_128)

        rec_high_32 = self.HF_Dec3(compress_high_hat_32)
        input_high_hat_32 = self.HF_Ref3(rec_high_32)

        img64 = self.Hy_Dec1(img64, input_high_hat_64)
        img128 = self.Hy_Dec2(img128, input_high_hat_128)
        img32 = self.Hy_Dec3(img32, input_high_hat_32)

        img64 = self.Decoder1(img64)
        img128 = self.Decoder2(img128)
        img32 = self.Decoder3(img32)

        img64 = self.Rev1(img64)
        img128 = self.Rev2(img128)
        img32 = self.Rev3(img32)

        return img128, img64, img32

    def load_Entropy_Model(self, component, path):
        checkpoint = torch.load(path, map_location=self.args.device)
        component.load_state_dict(checkpoint['Entropy_Model'])
        for param in component.parameters():
            param.requires_grad = False

    def load_HFM_Enc(self, component, path):
        checkpoint = torch.load(path, map_location=self.args.device)
        component.load_state_dict(checkpoint['HF_Enc'])
        for param in component.parameters():
            param.requires_grad = False

    def load_HFM_Dec(self, component, path):
        checkpoint = torch.load(path, map_location=self.args.device)
        component.load_state_dict(checkpoint['HF_Dec'])
        for param in component.parameters():
            param.requires_grad = False

    def Generate_LR(self, batch_images, upscale_factor=2):
        # Get the height and width from the input Tensor
        _, _, height, width = batch_images.shape

        # Calculate the new dimensions
        new_height = height // upscale_factor
        new_width = width // upscale_factor

        # Resize using PyTorch's F.interpolate
        lr_images = F.interpolate(batch_images, size=(new_height, new_width), mode='bicubic', align_corners=False)

        return lr_images

    def error_mask(self, tensor_rate, percent, error_rate=1 / 48):

        channel_sums = tensor_rate.view(tensor_rate.size(0), tensor_rate.size(1), -1).sum(dim=-1)

        k = int(channel_sums.size(1) * percent)

        _, topk_indices = torch.topk(channel_sums, k, dim=1)

        mask = torch.zeros_like(channel_sums, dtype=torch.int32)
        mask.scatter_(1, topk_indices, 1)

        B, C = mask.shape
        num_swap = max(1, int(C * error_rate)) if error_rate > 0 else 0

        for b in range(B):
            ones = (mask[b] == 1).nonzero(as_tuple=False).squeeze(1)
            zeros = (mask[b] == 0).nonzero(as_tuple=False).squeeze(1)

            swap_times = min(num_swap, len(ones), len(zeros))

            if swap_times > 0:
                swap_ones = ones[torch.randperm(len(ones))[:swap_times]]
                swap_zeros = zeros[torch.randperm(len(zeros))[:swap_times]]

                mask[b, swap_ones] = 0
                mask[b, swap_zeros] = 1

        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(tensor_rate)
        return mask

    def generate_mask_top_a_to_k(self, tensor_rate, percent):
        channel_sums = tensor_rate.view(tensor_rate.size(0), tensor_rate.size(1), -1).sum(dim=-1)
        total_channels = channel_sums.size(1)
        k = int(total_channels * percent)

        _, topk_indices = torch.topk(channel_sums, k + 2, dim=1)

        topk_indices = topk_indices[:, 2:]  # shape: [B, k]

        mask = torch.zeros_like(channel_sums, dtype=torch.int32).scatter_(1, topk_indices, 1)
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(tensor_rate)
        return mask

    def generate_mask(self, tensor_rate, percent):

        channel_sums = tensor_rate.view(tensor_rate.size(0), tensor_rate.size(1), -1).sum(dim=-1)

        k = int(channel_sums.size(1) * percent)

        _, topk_indices = torch.topk(channel_sums, k, dim=1)

        mask = torch.zeros_like(channel_sums, dtype=torch.int32).scatter_(1, topk_indices, 1)

        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(tensor_rate)
        return mask

    def random_generate_mask(self, tensor_rate, percent):

        channel_sums = tensor_rate.view(tensor_rate.size(0), tensor_rate.size(1), -1).sum(dim=-1)

        k = int(channel_sums.size(1) * percent)
        batch_size, num_channels = channel_sums.size(0), channel_sums.size(1)

        topk_indices = torch.stack([
            torch.randperm(num_channels, device=tensor_rate.device)[:k]
            for _ in range(batch_size)
        ])

        mask = torch.zeros_like(channel_sums, dtype=torch.int32).scatter_(1, topk_indices, 1)

        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(tensor_rate)
        return mask

    def visualize_all_feature_maps(self, tensor, title='All Feature Maps'):
        """
        Visualize all feature maps from a tensor of shape [B, C, H, W].

        Args:
            tensor (torch.Tensor): Input tensor of shape [B, C, H, W]
            title (str): Figure title
        """
        assert tensor.dim() == 4, "Tensor must be 4D [B, C, H, W]"

        feature_maps = tensor[0].detach().cpu().numpy()  # shape: [C, H, W]
        C = feature_maps.shape[0]

        num_cols = 8
        num_rows = math.ceil(C / num_cols)

        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        for i in range(C):
            plt.subplot(num_rows, num_cols, i + 1)
            fmap = feature_maps[i]
            plt.imshow(fmap, cmap='viridis')
            plt.title(f'Ch {i}', fontsize=8)
            plt.axis('off')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def visualize_feature_at(self, tensor, row=5, col=3, num_cols=8):

        assert tensor.dim() == 4, "Tensor must be 4D [B, C, H, W]"

        channel_idx = row * num_cols + col
        total_channels = tensor.shape[1]

        if channel_idx >= total_channels:
            print(f"Channel index out of range: {channel_idx} (total channels: {total_channels})")
            return

        feature = tensor[0, channel_idx].detach().cpu().numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(feature, cmap='viridis')
        plt.title(f'Feature Map at Row {row}, Col {col} (Ch {channel_idx})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

