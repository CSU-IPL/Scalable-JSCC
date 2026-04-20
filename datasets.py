
from torchvision import transforms, datasets
from Get_datasets import *
import torch.utils.data as data




NUM_DATASET_WORKERS = 10
SCALE_MIN = 0.75
SCALE_MAX = 0.95

def get_loader(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    Res_Train_high = "./datasets/" + args.train_dataset + "/train/" + str(args.train_resultion // 1)
    Res_Train_medium = "./datasets/" + args.train_dataset + "/train/" + str(args.train_resultion // 2)
    Res_Train_low = "./datasets/" + args.train_dataset + "/train/" + str(args.train_resultion // 4)

    Res_Val_high = "./datasets/" + args.test_dataset + "/val/" + str(args.test_resultion // 1)
    Res_Val_medium = "./datasets/" + args.test_dataset + "/val/" + str(args.test_resultion // 2)
    Res_Val_low = "./datasets/" + args.test_dataset + "/val/" + str(args.test_resultion // 4)

    train_dataset = MultiResolutionDataset(
        low_res_dir=Res_Train_high,
        medium_res_dir=Res_Train_medium,
        high_res_dir=Res_Train_low,
        transform=transform
    )

    test_dataset = MultiResolutionDataset(
        low_res_dir=Res_Val_high,
        medium_res_dir=Res_Val_medium,
        high_res_dir=Res_Val_low,
        transform=transform
    )
    train_loader = data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=args.BatchSize,
                                               shuffle=True,
                                               drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=args.BatchSize,
                                  shuffle=False)

    return train_loader, test_loader

