import torch
import torch.nn as nn
from model import recons_video_ori
from model import network_swinir_indep
from utils import utils
from torchvision.ops import DeformConv2d

def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    # load_flow_net = True
    load_recons_net = False
    # flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return CDVD_TSP(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_recons_net=load_recons_net, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device, args=args)


class CDVD_TSP(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda', args=None):
        super(CDVD_TSP, self).__init__()
        print("Creating CDVD-TSP Net")

        self.n_sequence = n_sequence
        self.device = device

        # assert n_sequence == 5, "Only support args.n_sequence=5; but get args.n_sequence={}".format(n_sequence)

        self.is_mask_filter = is_mask_filter
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 0
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        # self.flow_net = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.swin = network_swinir_indep.SwinIR(upscale=1,
                   in_chans=n_feat * 4,
                   img_size=args.patch_size//4,
                   window_size=args.window_size,
                   img_range=args.rgb_range,
                   depths=args.depths,
                   embed_dim=args.embed_dim,
                   num_heads=args.num_heads,
                   mlp_ratio=args.mlp_ratio,
                   resi_connection=args.resi_connection)
        self.recons_net = recons_video_ori.RECONS_VIDEO(in_channels=in_channels, n_sequence=5, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        # self.deformconv_nsf = DeformConv2d
        self.fusion_mid = nn.Conv2d(n_feat*4*2, n_feat*4, kernel_size=1, stride=1, padding=0)
        self.fusion_left = nn.Conv2d(n_feat * 4 * 2, n_feat * 4, kernel_size=1, stride=1, padding=0)
        self.fusion_right = nn.Conv2d(n_feat * 4 * 2, n_feat * 4, kernel_size=1, stride=1, padding=0)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def forward(self, x):  #, bm, label
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        # bm_list = [bm[:, i, :, :, :] for i in range(self.n_sequence)]
        left_pre_sharp_frame = x[:, self.n_sequence, :, :, :]
        right_sub_sharp_frame = x[:, self.n_sequence + 1, :, :, :]

        feature_left = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[0])))
        feature_mid = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[self.n_sequence//2])))
        feature_right = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[2])))
        feature_left_pre = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(left_pre_sharp_frame)))
        feature_right_sub = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(right_sub_sharp_frame)))

        trans_left = self.swin(feature_left, feature_left_pre, True)
        trans_right = self.swin(feature_right, feature_right_sub, True)

        fusion_left = torch.cat((trans_left, feature_left),dim=1)
        fusion_right = torch.cat((trans_right, feature_right), dim=1)
        fusion_left = self.fusion_left(fusion_left)
        fusion_right = self.fusion_right(fusion_right)

        trans_fea_l = self.swin(feature_mid, fusion_left, False)  #mid2neighbour:fea, feature_mid
        trans_fea_r = self.swin(feature_mid, fusion_right, False)
        fusion = torch.cat((trans_fea_l, trans_fea_r), dim=1)

        fusion = self.fusion_mid(fusion)
        out = self.recons_net.outBlock(self.recons_net.decoder_first(self.recons_net.decoder_second(fusion)))

        return out
