import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer


class Trainer_SWINT(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_SWINT, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-CDVD-TSP")
        # assert args.n_sequence == 5, \
        #     "Only support args.n_sequence=5; but get args.n_sequence={}".format(args.n_sequence)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.get_model().parameters(), **kwargs)

    def train(self):
        print("Now training")
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch # + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.

        for batch, (input, gt, bm, label, _) in enumerate(self.loader_train):

            input = input.to(self.device)
            # bm = bm.to(self.device)

            gt_list = [gt[:, i, :, :, :] for i in range(self.args.n_sequence)]
            gt = gt_list[self.args.n_sequence//2].to(self.device)
            # print(input.size(),bm.size(),label.size())
            out = self.model(input)  #, bm, label

            self.optimizer.zero_grad()
            loss = self.loss(out, gt)
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch #+ 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            total_PSNR_iter1 = 0.
            total_num = 0.
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, bm, label, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                input_center = input[:, self.args.n_sequence // 2, :, :, :]
                # bm = bm.to(self.device)
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                out = self.model(input)

                PSNR_iter1 = utils.calc_psnr(gt, out, rgb_range=self.args.rgb_range)
                total_PSNR_iter1 += PSNR_iter1
                total_num += 1
                PSNR = PSNR_iter1 #utils.calc_psnr(gt, recons_2_iter, rgb_range=self.args.rgb_range)
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    gt, input_center, out = utils.postprocess(gt, input_center, out,
                                                                                  rgb_range=self.args.rgb_range,
                                                                                  ycbcr_flag=False, device=self.device)
                    save_list = [gt, input_center, out]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR_iter1: {:.3f} PSNR_iter2: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                total_PSNR_iter1 / total_num,
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))  #
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))  #
