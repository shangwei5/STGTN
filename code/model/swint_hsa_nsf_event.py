import torch
import torch.nn as nn
from model import recons_video_ori
from model import network_swinir
from utils import utils
from model import SearchTransfer
import torch.nn.functional as F
from model.blocks import SEBlock

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
        self.swin = network_swinir.SwinIR(upscale=1,
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
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.conv_lv1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, stride=1, padding=0)
        self.conv_lv2 = nn.Conv2d(n_feat *2 * 2, n_feat*2, kernel_size=1, stride=1, padding=0)
        self.conv_lv3 = nn.Conv2d(n_feat *4* 2, n_feat *4, kernel_size=1, stride=1, padding=0)
        self.fusion = nn.Conv2d(n_feat*4*3, n_feat*4, kernel_size=1, stride=1, padding=0)
        self.event_in = nn.Sequential(
                nn.Conv2d(40, n_feat, kernel_size=5, stride=1, padding=5 // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feat, n_feat * 2, kernel_size=5, stride=2, padding=5 // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feat*2, n_feat * 2, kernel_size=5, stride=1, padding=5 // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=5, stride=2, padding=5 // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=5, stride=1, padding=5 // 2),
                nn.ReLU(inplace=True),
                SEBlock(n_feat * 4, 4)
        )
        # self.event_fusion = nn.Sequential(
        #     nn.Conv2d(n_feat * 4 *2, n_feat * 4, kernel_size=1, stride=1, padding=0),
        #     #
        # )
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def forward(self, x, event):  #, bm, label
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        # print(event.size())
        event = event[:, 0, :, :, :]

        left_pre_sharp_frame = frame_list[0]
        right_sub_sharp_frame = frame_list[-1]
        feature_mid = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[self.n_sequence//2])))
        fusion = feature_mid

        left_pre_lv1 = self.recons_net.inBlock(left_pre_sharp_frame)
        left_pre_lv2 = self.recons_net.encoder_first(left_pre_lv1)
        left_pre_lv3 = self.recons_net.encoder_second(left_pre_lv2)

        right_sub_lv1 = self.recons_net.inBlock(right_sub_sharp_frame)
        right_sub_lv2 = self.recons_net.encoder_first(right_sub_lv1)
        right_sub_lv3 = self.recons_net.encoder_second(right_sub_lv2)
        for i in range(self.n_sequence):
            if i != self.n_sequence // 2 - 1 and i != self.n_sequence // 2 + 1:
                # print(i)
                continue
            fea = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[i])))
            trans_fea = self.swin(feature_mid, fea)  #neighbour2mid:feature_mid, fea       mid2neighbour:fea, feature_mid
            fusion = torch.cat((fusion, trans_fea), dim=1)
            # print(out.size(), trans_fea.size())
        if self.n_sequence == 1:
            trans_fea = self.swin(feature_mid, feature_mid)  #neighbour2mid:feature_mid, fea       mid2neighbour:fea, feature_mid
            fusion = fusion + trans_fea
        fusion = self.fusion(fusion)

        #event fusion
        event_f = self.event_in(event)
        # fusion = self.event_fusion(torch.cat((fusion, event_f), dim=1))
        fusion = F.relu(fusion + event_f, True)

        left_pre_S, left_pre_T_lv3, left_pre_T_lv2, left_pre_T_lv1 = \
            self.SearchTransfer(fusion, left_pre_lv3, left_pre_lv1, left_pre_lv2, left_pre_lv3)
        right_sub_S, right_sub_T_lv3, right_sub_T_lv2, right_sub_T_lv1 = \
            self.SearchTransfer(fusion, right_sub_lv3, right_sub_lv1, right_sub_lv2, right_sub_lv3)

        left_v3 = self.conv_lv3(torch.cat((fusion, left_pre_T_lv3), dim=1)) * left_pre_S
        right_v3 = self.conv_lv3(torch.cat((fusion, right_sub_T_lv3), dim=1)) * right_sub_S
        fusion_lv3 = fusion + left_v3 + right_v3

        fusion_lv2 = self.recons_net.decoder_second(fusion_lv3)
        left_v2 = self.conv_lv2(torch.cat((fusion_lv2, left_pre_T_lv2), dim=1)) * F.interpolate(left_pre_S, scale_factor=2, mode='bicubic')
        right_v2 = self.conv_lv2(torch.cat((fusion_lv2, right_sub_T_lv2), dim=1)) * F.interpolate(right_sub_S, scale_factor=2, mode='bicubic')
        fusion_lv2 = fusion_lv2 + left_v2 + right_v2

        fusion_lv1 = self.recons_net.decoder_first(fusion_lv2)
        left_v1 = self.conv_lv1(torch.cat((fusion_lv1, left_pre_T_lv1), dim=1)) * F.interpolate(left_pre_S,
                                                                                                scale_factor=4,
                                                                                                mode='bicubic')
        right_v1 = self.conv_lv1(torch.cat((fusion_lv1, right_sub_T_lv1), dim=1)) * F.interpolate(right_sub_S,
                                                                                                  scale_factor=4,
                                                                                                  mode='bicubic')
        fusion_lv1 = fusion_lv1 + left_v1 + right_v1
        out = self.recons_net.outBlock(fusion_lv1)
        # out = self.recons_net.outBlock(self.recons_net.decoder_first(self.recons_net.decoder_second(fusion_lv3)))

        return out
