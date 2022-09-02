import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from model.swint_hsa_nsf_event import CDVD_TSP
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.nn as nn
from data.event_txt2npy_norm import txt2npy

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.n_seq = args.n_sequence
        self.size_must_mode = args.size_must_mode
        self.device = 'cuda'
        self.GPUs = args.n_GPUs

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = CDVD_TSP(in_channels=args.n_colors, n_sequence=self.n_seq, out_channels=args.n_colors, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda', args=args)
        self.net.load_state_dict(torch.load(self.model_path))  #, strict=False
        self.net = self.net.to(self.device)
        if args.n_GPUs > 1:
            self.net = nn.DataParallel(self.net, range(args.n_GPUs))
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()


    def infer(self):
        with torch.no_grad():

            videos = sorted(os.listdir(self.data_path))
            for v in videos:

                input_frames = sorted(glob.glob(os.path.join(self.data_path, v, "blurimage", "*")))
                event_frames = sorted(glob.glob(os.path.join(self.data_path, v, "Event", "*")))
                input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                event_seqs = self.gene_seq(event_frames, n_seq=self.n_seq)

                for in_seq, event_seq in zip(input_seqs, event_seqs):   #, bm_seq, label_seq   , bm_seqs, label_seqs
                    start_time = time.time()
                    filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                    # print(in_seq, gt_seq, event_seq)
                    # inputs = [imageio.imread(p) for p in in_seq]
                    inputs = [imageio.imread(p, as_gray=True) for p in in_seq]
                    in_tensor = self.numpy2tensor(inputs).to(self.device)
                    b, t, c, h, w = in_tensor.size()
                    events = [txt2npy(p, h, w).transpose(1,2,0) for p in event_seq[self.n_seq // 2:self.n_seq // 2+1]]

                    in_tensor = self.numpy2tensor(inputs).to(self.device)
                    event_tensor = self.numpy2tensor(events, 255.).to(self.device)
                    if h % self.size_must_mode != 0 or w % self.size_must_mode != 0:
                        in_tensor = F.pad(in_tensor, pad=[0, self.size_must_mode-w % self.size_must_mode, 0, self.size_must_mode-h % self.size_must_mode, 0, 0], mode='replicate')
                        event_tensor = F.pad(event_tensor, pad=[0, self.size_must_mode-w % self.size_must_mode, 0, self.size_must_mode-h % self.size_must_mode, 0, 0], mode='replicate')

                    preprocess_time = time.time()
                    # print(in_tensor.size(), event_tensor.size())
                    if self.GPUs ==1:
                        output= self.net(in_tensor, event_tensor)  #, bm_tensor, label_tensor
                    else:
                        output = self.forward_chop(in_tensor)
                    forward_time = time.time()
                    output_img = self.tensor2numpy(output)[:h, :w, :]
                    output_img = output_img.squeeze()

                    if self.save_image:
                        if not os.path.exists(os.path.join(self.result_path, v)):
                            os.mkdir(os.path.join(self.result_path, v))
                        imageio.imwrite(os.path.join(self.result_path, v, '{}.png'.format(filename)), output_img)
                        # imageio.imwrite(os.path.join(self.result_path, v, '{}_1.png'.format(filename)), output_img[:,:,0])
                        # imageio.imwrite(os.path.join(self.result_path, v, '{}_2.png'.format(filename)),
                        #                 output_img[:, :, 1])
                        # imageio.imwrite(os.path.join(self.result_path, v, '{}_3.png'.format(filename)),
                        #                 output_img[:, :, 2])
                    postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}-{} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, filename,
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))


    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[:half]
            img_list_temp.extend(img_list)
            img_list_temp.extend(img_list[-half:])
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            if img.ndim == 2:
                img = img[:,:, np.newaxis]
                # img = np.stack((img,) * 3, axis=-1)
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def forward_chop(self, *args, shave_h=20, shave_w=20, min_size=160000):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = 1  #self.opt['scale']
        n_GPUs = min(torch.cuda.device_count(), 4)
        # print(n_GPUs)
        args = [a.squeeze().unsqueeze(0) for a in args]

        # height, width
        h, w = args[0].size()[-2:]
        # print('len(args)', len(args))
        # print('args[0].size()', args[0].size())

        top = slice(0, h//2 + shave_h)
        bottom = slice(h - h//2 - shave_w, h)
        left = slice(0, w//2 + shave_h)
        right = slice(w - w//2 - shave_w, w)
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
                y = P.data_parallel(self.net.module, *x, range(n_GPUs))
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
                y = self.forward_chop(*p, shave_h=shave_h, shave_w=shave_w, min_size=min_size)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDVD-TSP-Inference')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', default=True, help='restore border images of video if true')

    parser.add_argument('--default_data', type=str, default='CVPR19',
                        help='quick test, optional: REDS, GOPRO')
    parser.add_argument('--data_path', type=str, default='../dataset/DVD/test',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
                        help='the path of pretrain model')
    parser.add_argument('--result_path', type=str, default='../infer_results',
                        help='the path of deblur result')
    args = parser.parse_args()

    if args.default_data == 'CED':
        args.data_path = '/media/r/sw/data/CED/test'
        args.model_path = '/media/r/sw/code/CDVD-TSP-master/experiment/swint_hsa_nsf_event/model/model_best.pt'
        args.result_path = '../infer_results/swint_hsa_nsf_ced'
        args.n_colors = 3
        args.n_sequence = 5
        args.patch_size = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 20
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.rgb_range = 1
        args.n_GPUs = 1
    elif args.default_data == 'CVPR19':
        args.data_path = '/mnt/disk10T/shangwei/data/deblur/Eextract-data'
        args.model_path = '../experiment/swint_hsa_nsf_event_grey/model/model_best.pt'
        args.result_path = '../infer_results/swint_hsa_nsf_cvpr19'
        args.n_colors = 1
        args.n_sequence = 5
        args.patch_size = 180
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 20
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.rgb_range = 1
        args.n_GPUs = 1


    Infer = Inference(args)
    Infer.infer()
