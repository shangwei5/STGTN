def set_template(args):
    if args.template == 'CDVD_TSP':
        args.task = "VideoDeblur"
        args.model = "CDVD_TSP"
        args.n_sequence = 5
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
    elif args.template == 'D2NET':
        args.task = "VideoDeblur"
        args.model = "D2NET"
        args.n_sequence = 3
        args.n_frames_per_video = 300
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 300
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
        args.epochs = 800
        args.batch_size = 16
        args.n_GPUs = 2
    elif args.template == 'D2NET_REDS':
        args.task = "VideoDeblur"
        args.model = "D2NET"
        args.n_sequence = 3
        args.n_frames_per_video = 300
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 5e-5
        args.lr_decay = 20
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.dir_data = '/mnt/disk10T/shangwei/data/deblur/REDS_8x_Random/train'
        args.dir_data_test = '/mnt/disk10T/shangwei/data/deblur/REDS_8x_Random/val'
        args.epochs = 50
        args.batch_size = 20
        args.n_GPUs = 4
        args.pre_train = '/mnt/disk10T/shangwei/code/CDVD-TSP-master/experiment/D2Net_REDS/model/model_best.pt'
    elif args.template == 'D2NET_Offical':
        args.task = "VideoDeblur"
        args.model = "D2NET_Offical"
        args.n_sequence = 5
        args.n_frames_per_video = 300
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.dir_data = '/mnt/disk10T_2/shangwei/data/deblur/GOPRO/train'
        args.dir_data_test = '/mnt/disk10T_2/shangwei/data/deblur/GOPRO/val'
        args.epochs = 500
        args.batch_size = 20
        args.n_GPUs = 4
        args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/D2Net_finetune/model/model_best.pt'
    elif args.template == 'D2NET_EVENT':
        args.task = "VideoDeblur"
        args.model = "D2NET_EVENT"
        args.n_sequence = 3
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.data_train = 'DVD_NFS_EVENT'
        args.data_test = 'DVD_NFS_EVENT'
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
        args.epochs = 500
        args.batch_size = 16
        args.n_GPUs = 2
        args.pre_train = '/media/r/sw/code/CDVD-TSP-master/experiment/D2Net_finetune/model/model_best.pt'
    elif args.template == 'SWINT':
        args.task = "VideoDeblur"
        args.model = "SWINT"
        args.n_sequence = 1
        args.patch_size = 200
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 5e-5
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.pre_train = '/mnt/disk10T/shangwei/code/CDVD-TSP-master/experiment/swint_neighbour2mid_seq1/model/model_best.pt'
    elif args.template == 'SWINT_HSA_NSF':
        args.task = "VideoDeblur"
        args.model = "SWINT_HSA_NSF"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 12
        args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/swint_neighbour2mid_seq3/model/model_best.pt'
    elif args.template == 'SWINT_HSA_NSF_Offical':
        args.task = "VideoDeblur"
        args.model = "SWINT_HSA_NSF_Offical"
        args.n_sequence = 5
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.dir_data = '/mnt/disk10T_2/shangwei/data/deblur/GOPRO/train'
        args.dir_data_test = '/mnt/disk10T_2/shangwei/data/deblur/GOPRO/val'
        args.batch_size = 12
        args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/swint_hsa_nsf/model/model_best.pt'
    elif args.template == 'SWINT_HSA_NSF_REDS':
        args.task = "VideoDeblur"
        args.model = "SWINT_HSA_NSF"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 5e-5
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 12
        args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/swint_hsa_nsf/model/model_best.pt'
        args.dir_data = '/mnt/disk10T_2/shangwei/data/deblur/REDS_8x_Random/train'
        args.dir_data_test = '/mnt/disk10T_2/shangwei/data/deblur/REDS_8x_Random/val'
    elif args.template == 'SWINT_HSA_NSF_EVENT':
        args.task = "VideoDeblur"
        args.model = "SWINT_HSA_NSF_EVENT"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 20
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 20
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.epochs = 50
        args.resi_connection = "1conv"
        args.data_train = 'DVD_EVENT'
        args.data_test = 'DVD_EVENT'
        args.batch_size = 12
        args.pre_train = '/media/r/sw/code/CDVD-TSP-master/experiment/swint_hsa_nsf_offical/model/model_best.pt'
        args.dir_data = '/media/r/sw/data/CED/train'
        args.dir_data_test = '/media/r/sw/data/CED/val'
        args.n_GPUs = 2
    elif args.template == 'SWINT_HSA_NSF_REDS_finetune':
        args.task = "VideoDeblur"
        args.model = "SWINT_HSA_NSF"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-6
        args.lr_decay = 2
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS_FUSION'
        args.data_test = 'DVD_NFS_FUSION'
        args.batch_size = 12
        args.epochs = 5
        args.pre_train = '/mnt/disk10T/shangwei/code/CDVD-TSP-master/experiment/swint_hsa_nsf_reds/model/model_best.pt'
        args.dir_data = '/mnt/disk10T/shangwei/data/deblur/BSDtest'
        args.dir_data2 = '/mnt/disk10T/shangwei/data/deblur/REDS_8x_Random/train'
        args.dir_data_test = '/mnt/disk10T/shangwei/data/deblur/BSDval'
    elif args.template == 'HSA_NSF':
        args.task = "VideoDeblur"
        args.model = "HSA_NSF"
        args.n_sequence = 1
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 20
        args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/swint_neighbour2mid_seq3/model/model_best.pt'
    elif args.template == 'SWINT_MASA_NSF':
        args.task = "VideoDeblur"
        args.model = "SWINT_MASA_NSF"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 8
        args.pre_train = '/media/r/sw/code/CDVD-TSP-master/experiment/swint_neighbour2mid_seq3/model/model_best.pt'
        args.n_GPUs = 2
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
    elif args.template == 'SWINT_NFS':
        args.task = "VideoDeblur"
        args.model = "SWINT_NFS"
        args.n_sequence = 3
        args.patch_size = 180
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 8
        args.n_GPUs = 2
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
    elif args.template == 'SWINT_NFS_INDEP':
        args.task = "VideoDeblur"
        args.model = "SWINT_NFS_INDEP"
        args.n_sequence = 3
        args.patch_size = 180
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 5e-5
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 12
        args.n_GPUs = 4
        # args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        # args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
        # args.epochs = 250
    elif args.template == 'SWINT_NFS_INDEP_PLUS':
        args.task = "VideoDeblur"
        args.model = "SWINT_NFS_INDEP_PLUS"
        args.n_sequence = 3
        args.patch_size = 180
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 2.5e-5
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 8
        args.n_GPUs = 4
        args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/swint_nfs_indep/model/model_best.pt'
        # args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        # args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
        # args.epochs = 250
    elif args.template == 'SWINT_NFS_INDEP_PLUS_DEFORM':
        args.task = "VideoDeblur"
        args.model = "SWINT_NFS_INDEP_PLUS_DEFORM"
        args.n_sequence = 3
        args.patch_size = 180   # 4*4*5  80
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 6
        args.n_GPUs = 2
        args.pre_train = '/media/r/sw/code/CDVD-TSP-master/experiment/swint_nfs_indep_plus/model/model_best.pt'
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
        # args.epochs = 250
    elif args.template == 'SWINT_NFS_INDEP_PLUS_REDS':
        args.task = "VideoDeblur"
        args.model = "SWINT_NFS_INDEP_PLUS"
        args.n_sequence = 3
        args.patch_size = 180
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 300
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 8
        args.n_GPUs = 4
        # args.pre_train = '/mnt/disk10T_2/shangwei/code/CDVD-TSP-master/experiment/swint_nfs_indep/model/model_best.pt'
        args.dir_data = '/mnt/disk10T_2/shangwei/data/deblur/REDS_8x_Random/train'
        args.dir_data_test = '/mnt/disk10T_2/shangwei/data/deblur/REDS_8x_Random/val'
        args.epochs = 800
    elif args.template == 'SWINT_FLOW_NFS':
        args.task = "VideoDeblur"
        args.model = "SWINT_FLOW_NFS"
        args.n_sequence = 3
        args.patch_size = 180
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 6
        args.n_GPUs = 2
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
    elif args.template == 'FLOW_SWINT_NFS':
        args.task = "VideoDeblur"
        args.model = "FLOW_SWINT_NFS"
        args.n_sequence = 3
        args.patch_size = 180
        args.n_frames_per_video = 300
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 12
        args.n_GPUs = 2
        args.dir_data = '/media/r/sw/data/GOPRO_Random/train'
        args.dir_data_test = '/media/r/sw/data/GOPRO_Random/val'
    elif args.template == 'BRNN':
        args.task = "VideoDeblur"
        args.model = "BRNN"
        args.n_sequence = 15
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.rnn_block = 6
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
    elif args.template == 'GPA':
        args.task = "VideoDeblur"
        args.model = "GPA"
        args.n_sequence = 5
        args.n_frames_per_video = 100
        args.n_feat = 64
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.n_neighbours = 2
        args.window_size = 13
        args.n_features = 256
        args.patch_size = 56
    elif args.template == 'MASA':
        args.task = "VideoDeblur"
        args.model = "MASA"
        args.n_sequence = 5
        args.n_frames_per_video = 100
        args.n_feat = 64
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.n_neighbours = 1
        args.patch_size = 256
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
