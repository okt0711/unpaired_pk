export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--FA 10 \
--TR 2.8 \
--act_type 'relu' \
--augment \
--batch_size 32 \
--beta1 0.5 \
--beta2 0.999 \
--continue_epoch 0 \
--data_root '' \
--deltt 6.505 \
--disp_step 10 \
--experiment_name '' \
--gpu_ids '0' \
--init_gain 1 \
--init_type 'xavier' \
--kinetic_model 'etofts' \
--lambda_cycle 10 \
--lambda_l1 0 \
--lambda_tv 0 \
--lr 1e-5 \
--lr_decay_iters 0 \
--lr_policy 'linear' \
--model 'DCE_supervised' \
--n_epochs 100 \
--n_epochs_decay 100 \
--n_time 65 \
--ndf 32 \
--ngf 128 \
--norm_type 'none' \
--patch_size 48 \
--r1 3.47 \
--save_epoch 50 \
--save_path '' \
--scale_ktrans 20 \
--scale_vp 40 \
--scale_ve 4 \
--test_epoch 200