device=0
dataset_path='./dataset'
CUDA_VISIBLE_DEVICES=${device} python fl_main.py --dataset cifar100 \
                                                 --numclass 5 \
                                                 --task_size 5 \
                                                 --img_size 32 \
                                                 --batch_size 128 \
                                                 --memory_size 2000 \
                                                 --epochs_local 20 \
                                                 --epochs_global 200 \
                                                 --dataset_path ${dataset_path}
