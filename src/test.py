from data import srraw_test
import sys
import os
import utility
import model
sys.path.append('..')
import utils
import scipy
import numpy as np
import PIL.Image as Image
from option import args
#testDataset = srraw_test.SRRAW_TEST(args, 'SRRAW')
from PIL import Image
from data import srraw
from matplotlib import pyplot as plt
import torch
testDataset = srraw.SRRAW(args, 'SRRAW', train=False)
print(len(testDataset))
psnrs = []
ssims = []
data_root = "/home/ibrain/git/PerceptualSimilarity/imgs/zoom/x"+str(args.scale[0])

if not os.path.exists(os.path.join(data_root, args.model)):
    os.makedirs(os.path.join(data_root, args.model))

##import model
if args.model != 'LR':
    torch.set_grad_enabled(False)
    checkpoint = utility.checkpoint(args)
    _model = model.Model(args, checkpoint)
    _model = _model.cuda()
    _model.eval()
for data in testDataset:
    lr = data[0]
    #print(lr.shape)
    hr = data[1][0]
    #print(hr.shape)
    print(data[2])
    if args.model == 'LR':
        lr = lr.numpy()
        hr = hr.numpy()
        lr = np.transpose(lr, (1,2,0))
        hr = np.transpose(hr, (1,2,0))
        sr = np.array(Image.fromarray(lr.astype(np.uint8)).resize(hr.shape[:2], Image.BICUBIC))
        #lr = scipy.misc.imresize(lr, hr.shape, interp='bicubic')
        #print(os.path.join(data_root, "lr", data[2]+'.png'))

        #print(lr)
        #print(hr)
    else:

        lr = lr.unsqueeze(0).cuda()
        hr = hr.unsqueeze(0).cuda()

        sr = _model(lr, 0)
        if isinstance(sr, tuple):
            g_slabel = sr[1]
            sr = sr[0]

            g_slabel = utility.quantize(g_slabel, args.rgb_range)
            g_slabel = g_slabel[0].detach().cpu().numpy()
            g_slabel = np.transpose(g_slabel, (1,2,0))
            g_slabel = g_slabel.astype(np.uint8)
            Image.fromarray(g_slabel).save(os.path.join(data_root, "RAT", data[2]+'.png'))

        sr = utility.quantize(sr, args.rgb_range)

        hr = hr[0].detach().cpu().numpy()
        sr = sr[0].detach().cpu().numpy()
        sr = np.transpose(sr, (1,2,0))
        hr = np.transpose(hr, (1,2,0))

    sr = sr.astype(np.uint8)
    Image.fromarray(sr).save(os.path.join(data_root, args.model, data[2]+'.png'))
    hr = hr.astype(np.uint8)
    Image.fromarray(hr).save(os.path.join(data_root, "hr", data[2]+'.png'))

    sr = sr / 255.0
    hr = hr / 255.0

    psnr = utils.calc_psnr(sr, hr, scale=int(args.scale[0]), rgb_range=1.)
    ssim = utils.calc_ssim(sr, hr)
    psnrs.append(psnr)
    ssims.append(ssim)
    print(psnr)
    #print(ssim)
    #print(psnr)

    # plt.subplot(121)
    # plt.imshow(lr.astype(np.uint8))
    #
    # plt.subplot(122)
    # plt.imshow(hr.astype(np.uint8))
    #
    # plt.show()
print(np.mean(np.array(ssims)))
print(np.mean(np.array(psnrs)))
print(len(testDataset))
