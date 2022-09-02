import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import torch.nn.parallel as P

class Trainer_SWINT_HSA_NSF(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_SWINT_HSA_NSF, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-CDVD-TSP")
        # assert args.n_sequence == 5, \
        #     "Only support args.n_sequence=5; but get args.n_sequence={}".format(args.n_sequence)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        # return optim.Adam([{"params": self.model.get_model().SearchTransfer.parameters()},
        #                    {"params": self.model.get_model().conv_lv1.parameters()},
        #                    {"params": self.model.get_model().conv_lv2.parameters()},
        #                    {"params": self.model.get_model().conv_lv3.parameters()},
        #                    {"params": self.model.get_model().recons_net.parameters(), "lr": 5e-5},
        #                    {"params": self.model.get_model().swin.parameters(), "lr": 2.5e-5},
        #                    {"params": self.model.get_model().fusion.parameters(), "lr": 5e-5}],
        #                   **kwargs)
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

        for batch, (input, gt, label, _) in enumerate(self.loader_train):

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
            for idx_img, (input, gt, label, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                input_center = input[:, self.args.n_sequence // 2, :, :, :]
                # bm = bm.to(self.device)
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                out = self.model(input)
                # out = self.forward_chop(input)

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

    def forward_chop(self, *args, shave=20, min_size=160000):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = 1  #self.opt['scale']
        n_GPUs = min(torch.cuda.device_count(), 4)
        # print(n_GPUs)
        args = [a.squeeze().unsqueeze(0) for a in args]

        # height, width
        h, w = args[0].size()[-2:]
        # print('len(args)', len(args))
        # print('args[0].size()', args[0].size())

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]
        # print('len(x_chops)', len(x_chops))
        # print('x_chops[0].size()', x_chops[0].size())

        y_chops = []
        if h * w < 6 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                # print(len(x))
                # print(x[0].size())
                y = P.data_parallel(self.model.get_model(), *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:

            # print(x_chops[0].size())
            for p in zip(*x_chops):
                # print('len(p)', len(p))
                # print('p[0].size()', p[0].size())
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1:
            y = y[0]

        return y
