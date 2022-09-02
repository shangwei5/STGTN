import os
from data import videodata_event_fusion_grey


class DVD_EVENT_FUSION_GREY(videodata_event_fusion_grey.VIDEODATA):
    def __init__(self, args, name='DVD', train=True):
        super(DVD_EVENT_FUSION_GREY, self).__init__(args, name=name, train=train)

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