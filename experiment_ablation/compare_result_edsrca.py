import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--e", type=int, default=199)
parser.add_argument("--root_path", type=str, default="/home/ibrain/git/pytorch_zoom/experiment_ablation")
ARGS = parser.parse_args()



apath = os.path.join(ARGS.root_path, 'EDSR_baseline_CAx4')
loss_log = torch.load(os.path.join(apath, 'loss_log.pt'))
psnr_log = torch.load(os.path.join(apath, 'psnr_log.pt'))
#print(psnr_log)

apath = os.path.join(ARGS.root_path, 'EDSRCA_MSAx4')
loss_b_log = torch.load(os.path.join(apath, 'loss_log.pt'))
psnr_b_log = torch.load(os.path.join(apath, 'psnr_log.pt'))

apath = os.path.join(ARGS.root_path, 'EDSRCA_SATx4')
loss_r_log = torch.load(os.path.join(apath, 'loss_SR_log.pt'))
psnr_r_log = torch.load(os.path.join(apath, 'psnr_log.pt'))

apath = os.path.join(ARGS.root_path, 'EDSRCA_SAT_MSAx4')
loss_s_log = torch.load(os.path.join(apath, 'loss_SR_log.pt'))
psnr_s_log = torch.load(os.path.join(apath, 'psnr_log.pt'))

#apath = os.path.join(ARGS.root_path, 'ssl_addfusion_reverse_rlossv0.2_x4')
print(len(loss_s_log), len(loss_r_log), len(loss_b_log), len(loss_log), ARGS.e)

e = min(len(loss_s_log), len(loss_r_log), len(loss_b_log), len(loss_log), ARGS.e)
print("Epoch:",e)
loss_log = loss_log[0:e]
psnr_log = psnr_log[0:e]
loss_b_log = loss_b_log[0:e]
psnr_b_log = psnr_b_log[0:e]
loss_r_log = loss_r_log[0:e]
psnr_r_log = psnr_r_log[0:e]
loss_s_log = loss_s_log[0:e]
psnr_s_log = psnr_s_log[0:e]


axis = np.linspace(1, e, e)
fig = plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(axis, loss_log.numpy(),color = 'black' ,label= r'RCAN$^{-}$')
plt.plot(axis, loss_b_log.numpy(),color = 'yellow' ,label= r'RCAN$^{-}$+MSA')

plt.plot(axis, loss_r_log.numpy(),color = 'cyan' ,label= r'RCAN$^{-}$+RAT' )
plt.plot(axis, loss_s_log.numpy(),color = 'red' ,label= r'RCAN$^{-}$+RAT+MSA')
plt.title("Training Loss Curve")
plt.legend()
plt.xlabel("Number of Epoches")
#plt.grid(True)

plt.subplot(122)
plt.plot(axis, savgol_filter(psnr_log.numpy().squeeze(),5,2),color = 'black' ,label= r'RCAN$^{-}$')
plt.plot(axis, savgol_filter(psnr_b_log.numpy().squeeze(),5,2),color = 'yellow' ,label= r'RCAN$^{-}$+MSA')

plt.plot(axis, savgol_filter(psnr_r_log.numpy().squeeze(),5,2),color = 'cyan' ,label= r'RCAN$^{-}$+RAT')
plt.plot(axis, savgol_filter(psnr_s_log.numpy().squeeze(),5,2),color = 'red' ,label= r'RCAN$^{-}$+RAT+MSA')

plt.title("PSNR on SRRAW Dataset")
plt.legend()
plt.xlabel("Number of Epoches")
#plt.grid(True)

plt.savefig("result_edsrca.pdf",bbox_inches = 'tight',pad_inches = 0)

#plt.show()
