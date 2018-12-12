import os
import time
import numpy as np
from scipy.misc import imread
import random
from threading import Thread
from scipy.io import loadmat


dataset_dir = '/home/kevin/HD2TB/Datasets/UCF101/'


def get_det_annotations():
    """
    Loads in all annotations for training and testing splits. This is assuming the data has been loaded in with resized
    frames (half in each dimension).
    :return: returns two lists (one for training and one for testing) with the file names and annotations. The lists
    contain tuples with the following content (file name, annotations), where annotations is a list of tuples with the
    form (start frame, end frame, label, bounding boxes).
    """
    f = loadmat(dataset_dir + 'UCF101_Annotations/trainAnnot.mat')
    f2 = loadmat(dataset_dir + 'UCF101_Annotations/testAnnot.mat')

    training_annotations = []
    for ann in f['annot'][0]:
        file_name = ann[1][0]

        sp_annotations = ann[2][0]
        annotations = []
        for sp_ann in sp_annotations:
            ef = sp_ann[0][0][0] - 1
            sf = sp_ann[1][0][0] - 1
            label = sp_ann[2][0][0] - 1
            bboxes = (sp_ann[3]/2).astype(np.int32)
            annotations.append((sf, ef, label, bboxes))

        training_annotations.append((file_name, annotations))

    testing_annotations = []
    for ann in f2['annot'][0]:
        file_name = ann[1][0]

        sp_annotations = ann[2][0]
        annotations = []
        for sp_ann in sp_annotations:
            ef = sp_ann[0][0][0] - 1
            sf = sp_ann[1][0][0] - 1
            label = sp_ann[2][0][0] - 1
            bboxes = (sp_ann[3] / 2).astype(np.int32)
            annotations.append((sf, ef, label, bboxes))

        testing_annotations.append((file_name, annotations))

    return training_annotations, testing_annotations


def get_video_det(video_dir, annotations, skip_frames=1, start_rand=True):
    """
    Loads in a video.

    :param video_dir: the directory of the video
    :param annotations: the annotations for that video
    :param skip_frames: number of frames which can be skipped
    :param start_rand: if True, then the first frame can be random (only useful if using skip_frames > 1). If False,
    then the first frame of the video will be the first frame of the video.
    :return: returns the video, bounding box annotations, and label
    """

    n_frames = len(os.listdir(video_dir))
    frame_start = 0

    im0 = imread(video_dir + ('frame_%d.jpg' % frame_start))
    h, w, ch = im0.shape
    video = np.zeros((n_frames, h, w, ch), dtype=np.uint8)
    video[0] = im0
    for f in range(n_frames):
        video[f] = imread(video_dir + ('frame_%d.jpg' % f))

    bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
    label = -1
    for ann in annotations:
        start_frame, end_frame, label = ann[0], ann[1], ann[2]
        for f in range(start_frame, min(n_frames, end_frame+1)):
            try:
                x, y, w, h = ann[3][f-start_frame]
                bbox[f, y:y+h, x:x+w, :] = 1
            except:
                print(start_frame, end_frame)
                print(video_dir)
                exit()

    if skip_frames == 1:
        return video, bbox, label

    skip_vid_frames, skip_bbox_frames = [], []
    if start_rand:
        start_frame = np.random.randint(0, skip_frames)
    else:
        start_frame = 0

    for f in range(start_frame, n_frames, skip_frames):
        skip_vid_frames.append(video[f:f+1])
        skip_bbox_frames.append(bbox[f:f+1])

    skip_vid = np.concatenate(skip_vid_frames, axis=0)
    skip_bbox = np.concatenate(skip_bbox_frames, axis=0)

    return skip_vid, skip_bbox, label


def get_clip_det(video, bbox, clip_len=8, any_clip=False):
    """
    Creates a clip from a video.


    :param video: The video
    :param bbox: The bounding box annotations
    :param clip_len: The length of the clip
    :param any_clip: If True, then any clip from the video can be used. If False only clips with bounding box
    annotations in all frames can be used.
    :return: returns the video clip and the bounding box annotations clip
    """
    if any_clip:
        n_frames, _, _, _ = video.shape
        start_loc = np.random.randint(0, n_frames-clip_len)
    else:
        frame_anns = np.sum(bbox, axis=(1, 2, 3))
        ann_locs = np.where(frame_anns > 0)[0]
        ann_locs = [i for i in ann_locs if (i+clip_len-1) in ann_locs]

        try:
            start_loc = ann_locs[np.random.randint(0, len(ann_locs))]
        except:
            start_loc = min(np.where(frame_anns > 0)[0][0], video.shape[0]-clip_len)

    return video[start_loc:start_loc+clip_len], bbox[start_loc:start_loc+clip_len]


def crop_clip_det(clip, bbox_clip, crop_size=(112, 112), shuffle=True):
    """
    Crops the clip to a given spatial dimension

    :param clip: the video clip
    :param bbox_clip: the bounding box annotations clip
    :param crop_size: the size which the clip will be cropped to
    :param shuffle: If True, a random cropping will occur. If False, a center crop will be taken.
    :return: returns the cropped clip and the cropped bounding box annotation clip
    """

    frames, h, w, _ = clip.shape
    if not shuffle:
        margin_h = h - crop_size[0]
        h_crop_start = int(margin_h/2)
        margin_w = w - crop_size[1]
        w_crop_start = int(margin_w/2)
    else:
        h_crop_start = np.random.randint(0, h - crop_size[0])
        w_crop_start = np.random.randint(0, w - crop_size[1])

    return clip[:, h_crop_start:h_crop_start+crop_size[0], w_crop_start:w_crop_start+crop_size[1], :] / 255., \
           bbox_clip[:, h_crop_start:h_crop_start+crop_size[0], w_crop_start:w_crop_start+crop_size[1], :]


# The data generator for training. Outputs clips, bounding boxes, and labels for the training split.
class UCF101TrainDataGenDet(object):
    def __init__(self, sec_to_wait=5, frame_skip=2):
        self.train_files, _ = get_det_annotations()
        self.sec_to_wait = sec_to_wait
        self.frame_skip = frame_skip
        self.frames_dir = dataset_dir + 'UCF101_Frames/'

        np.random.seed(None)
        random.shuffle(self.train_files)

        self.data_queue = []

        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()

        print('Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.train_files:
            while len(self.data_queue) >= 600:
                time.sleep(1)
            vid_name, anns = self.train_files.pop()
            clip, bbox_clip, label = get_video_det(self.frames_dir + vid_name + '/', anns, skip_frames=self.frame_skip, start_rand=True)
            clip, bbox_clip = get_clip_det(clip, bbox_clip, any_clip=False)
            clip, bbox_clip = crop_clip_det(clip, bbox_clip, shuffle=True)
            self.data_queue.append((clip, bbox_clip, label))
        print('Loading data thread finished')

    def get_batch(self, batch_size=5):
        while len(self.data_queue) < batch_size and self.train_files:
            print('Waiting on data')
            time.sleep(self.sec_to_wait)

        batch_size = min(batch_size, len(self.data_queue))
        batch_x, batch_bbox, batch_y = [], [], []
        for i in range(batch_size):
            vid, bbox, label = self.data_queue.pop(0)
            batch_x.append(vid)
            batch_bbox.append(bbox)
            batch_y.append(label)

        return batch_x, batch_bbox, batch_y

    def has_data(self):
        return self.data_queue != [] or self.train_files != []


# The data generator for testing. Outputs clips, bounding boxes, and labels for the testing split.
class UCF101TestDataGenDet(object):
    def __init__(self, sec_to_wait=5, frame_skip=1):
        _, self.test_files = get_det_annotations()
        self.n_videos = len(self.test_files)
        self.sec_to_wait = sec_to_wait
        self.skip_frame = frame_skip
        self.frames_dir = dataset_dir + 'UCF101_Frames/'

        self.video_queue = []
        self.videos_left = self.n_videos
        self.data_queue = []

        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()

        print('Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.test_files:
            while len(self.data_queue) >= 50:
                time.sleep(1)
            vid_name, anns = self.test_files.pop(0)
            clip, bbox_clip, label = get_video_det(self.frames_dir + vid_name + '/', anns, skip_frames=self.skip_frame, start_rand=False)
            clip, bbox_clip = crop_clip_det(clip, bbox_clip, shuffle=False)
            self.data_queue.append((clip, bbox_clip, label))
        print('Loading data thread finished')

    def get_next_video(self):
        while len(self.data_queue) == 0:
            print('Waiting on data')
            time.sleep(self.sec_to_wait)

        return self.data_queue.pop(0)

    def has_data(self):
        return self.data_queue != [] or self.test_files != []


