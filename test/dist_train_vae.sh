CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 test/test_dist_train_vae.py --resume --batch_size 8 --epoch 35 --autocast