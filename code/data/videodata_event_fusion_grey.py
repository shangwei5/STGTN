import os
import glob
import numpy as np
import imageio
import torch
import torch.utils.data as data
import utils.utils as utils
from data.event_txt2npy_norm import txt2npy
from skimage.transform import resize

class VIDEODATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequence
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data, args.dir_data2)
        else:
            self._set_filesystem(args.dir_data_test, None)

        self.images_gt, self.images_input, self.images_event = self._scan()   #

        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_gt, self.data_input, self.data_label = self._load(self.images_gt, self.images_input, self.images_label)   #

    def _set_filesystem(self, dir_data, dir_data2=None):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.apath2 = dir_data2
        self.dir_gt = os.path.join(self.apath, 'gt')
        self.dir_input = os.path.join(self.apath, 'blur')
        self.dir_event = os.path.join(self.apath, 'Event')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)
        print("DataSet event path:", self.dir_event)
        if self.apath2 is not None:
            self.dir_gt2 = os.path.join(self.apath2, 'gt')
            self.dir_input2 = os.path.join(self.apath2, 'blur')
            self.dir_event2 = os.path.join(self.apath2, 'Event')
            print("DataSet GT path:", self.dir_gt, self.dir_gt2)
            print("DataSet INPUT path:", self.dir_input, self.dir_input2)
            print("DataSet label path:", self.dir_event, self.dir_event2)
        else:
            print("DataSet GT path:", self.dir_gt)
            print("DataSet INPUT path:", self.dir_input)
            print("DataSet label path:", self.dir_event)


    def _scan(self):
        vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        vid_event_names = sorted(glob.glob(os.path.join(self.dir_event, '*')))
        if self.apath2 is not None:
            vid_gt_names.extend(sorted(glob.glob(os.path.join(self.dir_gt2, '*'))))
            vid_input_names.extend(sorted(glob.glob(os.path.join(self.dir_input2, '*'))))
            vid_event_names.extend(sorted(glob.glob(os.path.join(self.dir_event2, '*'))))
        assert len(vid_gt_names) == len(vid_input_names) == len(vid_event_names), "len(vid_gt_names) must equal len(vid_input_names)"  #

        images_gt = []
        images_input = []
        images_event = []

        for vid_gt_name, vid_input_name, vid_event_name in zip(vid_gt_names, vid_input_names, vid_event_names):  #
            if self.train:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))[:self.args.n_frames_per_video]
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.args.n_frames_per_video]
                event_dir_names = sorted(glob.glob(os.path.join(vid_event_name, '*')))[:self.args.n_frames_per_video]
            else:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
                event_dir_names = sorted(glob.glob(os.path.join(vid_event_name, '*')))
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            images_event.append(event_dir_names)
            self.n_frames_video.append(len(gt_dir_names))

        return images_gt, images_input, images_event

    def _load(self, images_gt, images_input, images_label):  #
        data_input = []
        data_gt = []
        # video_num, single_video_frame_num, h,w,c

        n_videos = len(images_gt)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = np.array([imageio.imread(hr_name) for hr_name in images_gt[idx]])
            inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
            data_input.append(inputs)
            data_gt.append(gts)

        return data_gt, data_input, images_label

    def __getitem__(self, idx):
        if self.args.process:
            inputs, gts, labels, filenames = self._load_file_from_loaded_data(idx)   #
        else:
            inputs, gts, events, filenames = self._load_file(idx)   #
        # print(events.shape)
        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq)]
        inputs_concat = np.concatenate(inputs_list, axis=2)
        gts_list = [gts[i, :, :, :] for i in range(self.n_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)
        events_list = [events[i, :, :, :] for i in range(self.n_seq-2)]  #1
        events_concat = np.concatenate(events_list, axis=2)
        # print(events_concat.shape)
        inputs_concat, gts_concat, events_concat = self.get_patch(inputs_concat, gts_concat, events_concat, self.args.size_must_mode)  #
        inputs_list = [inputs_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq)]  #2
        gts_list = [gts_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq)]
        # print(events.shape, events_concat.shape)
        events_list = [events_concat[:, :, i * 40:(i + 1) * 40] for i in range(self.n_seq-2)]   #1

        inputs = np.array(inputs_list)
        gts = np.array(gts_list)
        events = np.array(events_list)

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        event_tensors = utils.eventnp2Tensor(*events)

        return torch.stack(input_tensors), torch.stack(gt_tensors), torch.stack(event_tensors),filenames  #

    def __len__(self):
        if self.train:
            return self.num_frame * self.repeat
        else:
            return 30  #self.num_frame

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        # f_events = self.images_event[video_idx][frame_idx+self.n_seq//2:frame_idx + self.n_seq//2+1]
        f_events = self.images_event[video_idx][frame_idx + self.n_seq // 2-1:frame_idx + self.n_seq // 2 + 2]
        # print(f_gts, f_inputs, f_events)
        h,w = imageio.imread(f_gts[0], as_gray=True).shape
        gts = np.array([resize(imageio.imread(hr_name, as_gray=True), (180,240)) for hr_name in f_gts])[:,:,:,np.newaxis]
        inputs = np.array([resize(imageio.imread(lr_name, as_gray=True), (180,240)) for lr_name in f_inputs])[:,:,:,np.newaxis]

        events = np.array([resize(txt2npy(event_name,h,w).transpose(1,2,0), (180,240)) for event_name in f_events])
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_inputs]
        # print(gts.shape, inputs.shape, events.shape)
        return inputs, gts, events, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        gts = self.data_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        # events = self.data_event[video_idx][frame_idx:frame_idx + self.n_seq]
        labels = self.data_label[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]]

        return inputs, gts, labels, filenames

    def get_patch(self, input, gt, event, size_must_mode=1):   #
        if self.train:
            input, gt, event = utils.get_patch(input, gt, event, patch_size=self.args.patch_size)   #
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt, event = input[:new_h, :new_w, :], gt[:new_h, :new_w, :], event[:new_h, :new_w, :]
            if not self.args.no_augment:
                input, gt, event = utils.data_augment(input, gt, event)   #
        else:
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            # print(input.shape,gt.shape,event.shape)
            input, gt, event = input[:new_h, :new_w, :], gt[:new_h, :new_w, :], event[:new_h, :new_w, :]
        return input, gt, event
