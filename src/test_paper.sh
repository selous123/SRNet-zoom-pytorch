python main.py --model EDSR --scale 4 --patch_size 256  --data_train SRRAW --data_test SRRAW --dir_data /store/dataset/SR/train_data --n_colors 3  --save_results --batch_size 16 --save_gt --labels HR+Diff --attn --save EDSR_baseline_CA --desc "edsr with channel attention" --epochs 200