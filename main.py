
from datasets import *
from model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

RED = "\033[31m"
RESET = "\033[0m"

class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.device = self.args.device
        self.gnet = model
        self.batch = self.args.BatchSize
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = 0
        self.MSELoss = nn.MSELoss()
        self.lr = 1e-4
        self.best_psnr = 0.
        if self.args.load_model:
            if not os.path.exists(self.args.save_path):
                print("No params")
                exit(-1)
            else:
                param_dict = torch.load(self.args.save_path, map_location=args.device)
                self.gnet.load_state_dict(param_dict["gnet_dict"])
                print(f"{RED}Loaded params:{RESET} " + (self.args.save_path))
        self.gnet.to(self.device)
        self.optimizer_gnet = torch.optim.AdamW(self.gnet.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                                                 eps=1e-08, weight_decay = args.wd, amsgrad=False)

    @staticmethod
    def calculate_psnr(img1, img2):
        mse = ((img1 - img2) ** 2).view(img1.size(0), -1).mean(dim=1)
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.mean()

    def train(self, epoch):
        self.gnet.train()
        train_loss_g = 0.
        train_loss_all_g = 0.
        psnr = 0.
        start = time.time()
        total = 0
        for i, (img128, img64, img32) in enumerate(self.train_loader):

            img128 = img128.to(self.device)
            img64 = img64.to(self.device)
            img32 = img32.to(self.device)


            rec_img_128, rec_img_64, rec_img_32 = self.gnet(img128)

            loss_128 = self.MSELoss(rec_img_128, img128)
            loss_64 = self.MSELoss(rec_img_64, img64)
            loss_32 = self.MSELoss(rec_img_32, img32)

            loss_g = 0.4 * loss_32 + 0.6 * loss_64 + 0.9 * loss_128

            self.optimizer_gnet.zero_grad()

            loss_g.backward(retain_graph=True)
            self.optimizer_gnet.step()

            train_loss_g += loss_g.item()
            train_loss_all_g += loss_g.item()
            psnr += (self.calculate_psnr(rec_img_32, img32).item() + self.calculate_psnr(rec_img_64, img64).item())  / 2

            total += 1

            if (i+1) % self.args.interval == 0:
                end = time.time()
                print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} gnet_loss:{:.5f} train_set_psnr:{:.4f}".format(
                    epoch, (i+1)*100/len(self.train_loader), end-start,
                    train_loss_g/self.args.interval, psnr/total
                ))
                train_loss_g = 0.

    def val(self, epoch):
        self.gnet.eval()
        if self.args.is_training:
            print("Test start...")
        val_loss = 0.
        psnr_32 = 0.
        psnr_64 = 0.
        psnr_128 = 0.
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, (img128, img64, img32) in enumerate(self.val_loader):
                img128 = img128.to(self.device)
                img64 = img64.to(self.device)
                img32 = img32.to(self.device)

                rec_img_128, rec_img_64, rec_img_32 = self.gnet(img128)

                psnr_32 += self.calculate_psnr(rec_img_32, img32).item()
                psnr_64 += self.calculate_psnr(rec_img_64, img64).item()
                psnr_128 += self.calculate_psnr(rec_img_128, img128).item()
                total += 1

            mpsnr = ((psnr_32 + psnr_64 + psnr_128) / 3) / total
            psnr_32_pri = psnr_32 / total
            psnr_64_pri = psnr_64 / total
            psnr_128_pri = psnr_128 / total
            end = time.time()
            if self.args.is_training:
                print("Test finished!")
                print("[Epoch]: {} time:{:.2f} loss:{:.5f} test_set_psnr_32:{:.2f} test_set_psnr_64:{:.2f}  test_set_psnr_128:{:.2f}".format(
                    epoch, end - start, val_loss / len(self.val_loader), psnr_32_pri, psnr_64_pri, psnr_128_pri
                ))
            else:
                print(str(round(psnr_32_pri, 2)) + ",", end="")
                print(str(round(psnr_64_pri, 2)) + ",",end="")
                print(str(round(psnr_128_pri, 2)) + ",", end="")
                print()
            # if mpsnr > self.best_psnr and self.args.is_training:
            if self.args.is_training:
                self.best_psnr = mpsnr
                print("Save params to {}".format(self.args.save_path))
                param_dict = {
                    "gnet_dict": self.gnet.state_dict(),
                }
                torch.save(param_dict, self.args.save_path)

def main():
    args = args_parser()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda:0':
        print(f'{RED}Device:{RESET}',args.device)
        torch.backends.cudnn.benchmark = True

    cfg_dict = get_cfg_by_ratio(ratio=args.ratio)

    model = SemanticComm(args, cfg_dict).to(args.device)

    train_loader, test_loader = get_loader(args)

    args.load_model = True
    args.is_training = False

    t = Trainer(args, model, train_loader, test_loader)

    if args.is_training == True:
        print(f"{RED}Training model:{RESET}", args.save_path)
        print(f"{RED}Training channel type:{RESET}", args.channel_type)
        print(f"{RED}Training dataset:{RESET}", args.test_dataset+
              "(3×"+str(args.train_resultion)+"×"+str(args.train_resultion)+")")

        for epoch in range(t.epoch, t.epoch + args.numepoch):
            t.train(epoch)
            t.val(epoch)

    else:
        print(f"{RED}Testing model:{RESET}", args.save_path)
        print(f"{RED}Test channel type:{RESET}", args.channel_type)
        print(f"{RED}Test dataset:{RESET}", args.test_dataset+
              "(3×"+str(args.test_resultion)+"×"+str(args.test_resultion)+")")

        if args.channel_type == 'awgn':
            args.random_number = 0 / 48
            for j in range(0, 3):
                args.csnr = 3
                for i in range(0, 6):
                    t.val(epoch="val")
                    args.csnr += 2
                print()
                args.random_number += 3 / 48

        elif args.channel_type == 'rayleigh':
            # test fading coefficient list
            # test_h_list = torch.tensor([1.2616, 1.1584, ..., 1.00])

            for k in range(len(test_h_list)):
                model.channel.h = test_h_list[k]
                print(f"{RED}Test slow Fading coefficient:{RESET}" + "h={:.4f}".format(model.channel.h.item()))
                args.random_number = 0 / 48
                for j in range(0, 3):
                    args.csnr = 3
                    for i in range(0, 6):
                        t.val(epoch="val")
                        args.csnr += 2
                    print()
                    args.random_number += 3 / 48

if __name__ == '__main__':
    main()


