import os
import glob
import random
import pickle

from data import common
#import common
import numpy as np
import imageio
import torch
import torch.utils.data as data
import sys
sys.path.append("..")
from option import args
sys.path.append("../..")
import utils
from PIL import Image

class SRRAW(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):

        ## data range split train and val dataset
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]


        self.begin, self.end = list(map(lambda x: int(x), data_range))

        self.benchmark = benchmark
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.input_large = False
        self.scale = args.scale
        self.idx_scale = 0

        self.n_colors = args.n_colors

        self._set_filesystem(args.dir_data)
        #print(self.n_colors)
        #exit(0)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_label, list_lr = self._scan()


        #exit(0)

        #print(path_bin)

        if args.ext.find('img') >= 0:
            self.images_label, self.images_lr = list_label, list_lr

        elif args.ext.find('sep') >= 0:

            for si, _ in enumerate(self.dir_label):
                os.makedirs(
                    self.dir_label[si].replace(self.apath, path_bin),
                    exist_ok=True
                )

            self.images_label, self.images_lr = [[] for _ in range(len(self.dir_label))], [[] for _ in self.scale]

            for si, s in enumerate(list_label):
                for h in s:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_label[si].append(b)
                    self._check_and_load(args.ext, h, b, verbose=True)


            if self.n_colors == 3:
                os.makedirs(
                    self.dir_lr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(args.ext, l, b, verbose=True)
            else:
                #ARW文件路径
                for i, ll in enumerate(list_lr):
                    for l in ll:
                        self.images_lr[i].append(l)

        self.images_hr = self.images_label[0]
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            #print("n_images:",n_images,len(args.data_train),n_patches)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        #2. list the file names in directory
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_label[0], '*' + self.ext[0]))
        )

        # names_edge = []
        # names_diff = []
        # for name_hr in names_hr:
        #     name_edge = name_hr.replace(self.dir_label[0], self.dir_label[1])
        #     names_edge.append(name_edge)
        #     name_diff = name_hr.replace(self.dir_hr, self.dir_diff)
        #     names_diff.append(name_diff)

        # names_label = [[sorted(
        #     glob.glob(os.path.join(dir, '*' + self.ext[0]))
        #     )] for dir in self.dir_label]
        names_label = [[] for _ in range(len(self.dir_label))]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.dir_label):
                #print(s)
                names_label[si].append(os.path.join(
                    s, '{}{}'.format(
                        filename, self.ext[0]
                    )
                ))

        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                        filename, self.ext[1]
                    )
                ))

        # split train and val dataset

        names_label = [n[self.begin - 1:self.end] for n in names_label]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_label, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name, 'X'+str(self.scale[0]), 'train')


        self.dir_label = [os.path.join(self.apath, label) for label in args.labels.split('+')]
        # dir_hr = os.path.join(self.apath, 'HR')
        # ##1.Add File Path into _set_filesystem
        # dir_edge = os.path.join(self.apath, 'Edge')
        # dir_diff = os.path.join(self.apath, 'Diff')
        #
        #  = [dir_hr, dir_edge, dir_diff]
        #print(self.dir_label)

        if self.n_colors == 4:
            self.dir_lr = os.path.join(self.apath, 'ARW')
            self.ext = ('.png', '.npy')
        elif self.n_colors == 3:
            self.dir_lr = os.path.join(self.apath, 'LR')
            self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, labels, filename = self._load_file(idx)
        pair = self.get_patch(lr, labels)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        #lr, hr, edge, diff
        #return pair_t[0], pair_t[1],pair_t[2], pair_t[3], filename
        return pair_t[0], pair_t[1:], filename
    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        fs_label = [image_label[idx] for image_label in self.images_label]

        #f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_lr))
        if self.args.ext == 'img':
            hr = imageio.imread(f_hr)
            labels = [[imageio.imread(f_label)] for f_label in fs_label]

            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            labels = []
            for f_label in fs_label:
                with open(f_label, 'rb') as _f:
                    label = pickle.load(_f)
                    labels.append(label)
            if self.n_colors == 3:
                with open(f_lr, 'rb') as _f:
                    lr = pickle.load(_f)
            elif self.n_colors == 4:
                lr = np.load(f_lr)

        return lr, labels, filename

    def get_patch(self, lr, labels):

        data = []
        data.append(lr)
        for label in labels:
            data.append(label)
        # LR, HR, labels

        scale = self.scale[self.idx_scale]
        if self.n_colors == 4:
            scale = scale * 2

        #print(scale)
        #print(self.args.test_patch_size)
        if self.train:
            data = common.get_patch(
                *data,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
        else:
            data = common.get_center_patch(
                *data,
                patch_size=self.args.test_patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )

        if self.train and not self.args.no_augment:
            data = common.augment(*data)
        #print(data[0].shape)
        #print(data[1].shape)
        return data

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

if __name__=="__main__":
    ## test dataset
    ## 读取 ARW 和 HR image 图像对
    ## 读取 LR 和 HR image 图像对

    dataset = SRRAW(args, name='SRRAW', train=False)
    print(len(dataset))
    for i in range(len(dataset)):
        lr = dataset[i][0]
        #print(lr)
        hr = dataset[i][1]
        print("index:%d"%i)
    hr = hr[0]
    #print(hr)
    #print(lr.shape)
    #print(hr.shape)

    #print(lr.shape)
    lr = lr.numpy()
    lr = np.transpose(lr, (1,2,0))
    hr = hr.numpy()
    hr = np.transpose(hr, (1,2,0))
    aligned_image = Image.fromarray(np.uint8(utils.clipped(lr)))
    aligned_image = aligned_image.resize((hr.shape[1], hr.shape[0]), Image.ANTIALIAS)
    lr = np.array(aligned_image)
    #print(hr)
    hr = hr / 255.0
    lr = lr / 255.0
    vis = True
    ## Visualization Results.
    if vis:
        min_img_t = np.abs(hr - lr)
        min_img_t_scale = (min_img_t - np.min(min_img_t)) / (np.max(min_img_t) - np.min(min_img_t))

        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.imshow(lr)

        plt.subplot(222)
        plt.imshow(hr)

        plt.subplot(223)
        plt.imshow(min_img_t_scale)

        # plt.subplot(224)
        # plt.imshow(wb_rgb)

        plt.show()
