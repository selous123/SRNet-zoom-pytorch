## in your local host seeing data_train
--dir_data /store2/dataset/SR/train_data
--wb_root /store2/dataset/SR/zoom/train/


## ibrain server
--dir_data /store/dataset/SR/train_data
--wb_root /store/dataset/zoom/train

## test
python main.py --model EDSR --scale 4  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train /home/ibrain/git/EDSR-PyTorch/experiment/model/EDSR_x4.pt --test_only --save_results
python main.py --model EDSR --scale 4  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train /home/ibrain/git/EDSR-PyTorch/experiment/model/EDSR_x4.pt --test_only --save_results

python main.py --model EDSR --scale 4 --pre_train ../models/edsr_baseline_x4-6b446fab.pt --test_only --save_results


## train baseline
python main.py --model EDSR --scale 4 --patch_size 256 --save edsr_baseline_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results
## train edsr in paper
python main.py --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --patch_size 256 --save edsr_paper_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --save_gt

## train baseline with epoch 300
python main.py --model EDSR --scale 4 --patch_size 256 --save edsr_baseline_x4_dynamic_lr --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --save_gt
## train baseline with epoch 500
python main.py --model EDSR --scale 4 --patch_size 256 --save edsr_baseline_x4_dynamic_lr_epoch500 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --save_gt
##debug SAN
python main.py --model SAN --scale 4 --patch_size 256 --save san_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --batch_size 8
## debug RCAN
python main.py --model RCAN --scale 4 --patch_size 256 --save rcan_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --batch_size 16

## debug srresnet
## ARW data
python main.py --model SRResNet --scale 4 --patch_size 256 --save srrsenet_x4_ARW --data_train SRRAW --data_test SRRAW --n_colors 4 --save_results --batch_size 16 --save_gt --labels HR --rgb_range 1
python main.py --model EDSR --scale 4 --patch_size 256 --save EDSR_x4_ARW --data_train SRRAW --data_test SRRAW --n_colors 4 --save_results --batch_size 16 --labels HR --rgb_range 1 --loss 100.0*L1
## PNG data
python main.py --model SRResNet --scale 4 --patch_size 256 --save srrsenet_x4 --data_train SRRAW --data_test SRRAW --n_colors 3 --save_results --batch_size 16


## debug our ssl model, improve 1 point in psnr
python main.py --model SSL --scale 4 --patch_size 256  --data_train SRRAW --data_test SRRAW --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR+Diff --save ssl_addfushion_reverse_x4
## debug r-loss
python main.py --model SSL --scale 4 --patch_size 256  --data_train SRRAW --data_test SRRAW --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR+Diff --save ssl_addfushion_reverse_rloss_x4 --loss_rel 1.0*RNLLoss
## debug r-loss v0.4
python main.py --model SSL --scale 4 --patch_size 256  --data_train SRRAW --data_test SRRAW --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR+Diff --save ssl_addfushion_reverse_rlossv0.4_x4 --loss_rel 1.0*RLoss
## debug attention model
python main.py --model SSL --scale 4 --patch_size 256  --data_train SRRAW --data_test SRRAW --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR+Diff --attn --save ssl_addfushion_reverse_x4_add_attn



## X4 model
python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale 0.1 --scale 4 --patch_size 256  --data_train SRRAW --data_test SRRAW --dir_data /store/dataset/SR/train_data --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR  --save EDSRPx4 --desc "EDSR paper 4X model" --data_range '1-1300/1301-1314'

## X8 model
python main.py --model SRResNet --scale 8 --patch_size 256  --data_train SRRAW --data_test SRRAW --dir_data /store/dataset/SR/train_data --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR --save SRResnetx8 --desc "srresnet x8" --epochs 200 --data_range '1-437/438-447'



##test script
python main.py --label HR --model EDSR --scale 4 --pre_train /store/git/pytorch_zoom/experiment_ablation/EDSR_baselinex4/model/model_best.pt --data_test SRRAW --test_only --dir_data /store2/dataset/SR/train_data --n_colors 3  --save_results --save_gt --save test_EDSRX4 --desc "test rdsr model on srraw test dataset"
python main.py --label HR --model SSL --scale 4 --pre_train /store/git/pytorch_zoom/experiment/EDSRCA_SAT_MSAx4/model/model_best.pt --data_test SRRAW_TEST --test_only --dir_data /store2/dataset/SR/train_data --n_colors 3  --save_results --save_gt --save SFNetX4_test --desc "test SFNet model on srraw test dataset" --attn


python test.py --label HR --desc "test bicubic psnr" --scale 8 --data_range "1-437/438-447"
python test.py --label HR --desc "test bicubic psnr" --dir_data /store/dataset/SR/train_data  --scale 4 --data_range "1-1300/1301-1314"
python test.py  --desc "bicubic test"  --dir_data /store/dataset/SR/train_data --scale 8 --data_range "1-482/483-497"  --pre_train  ~/git/pytorch_zoom/experiment/RCANx8_1/model/model_best.pt --model RCAN
