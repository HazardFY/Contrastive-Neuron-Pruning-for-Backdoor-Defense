import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
    parser.add_argument("--GPU_num", type=int, default=1)
    # path
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--classes_num",type=int,default=10)
    parser.add_argument("--data_root", type=str, default="./data/cifar10/")
    parser.add_argument('--test_model', type=str, default='0')
    # name
    # parser.add_argument("--save_dir", type=str, default="experiments")
    # parser.add_argument("--experiment_name", type=str, default="poisoned")
    parser.add_argument("--attack_method",type=str,default='badnet')
    # parser.add_argument("--defense_method",type=str,default='finetuning')
    parser.add_argument("--arch",type=str,default='resnet18')
    # hyparameters
    parser.add_argument("--target_label", type=int, default=3)
    parser.add_argument("--poisoned_rate", type=float, default=0.1)
    parser.add_argument("--benign_training", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--schedule", type=list, default=[50,100,150,180])
    parser.add_argument("--epochs", type=int, default=200)
    # Backdoor Box settings
    parser.add_argument("--poisoned_transform_train_index", type=int, default=0) # index of backdoor in training transform
    parser.add_argument("--poisoned_transform_test_index", type=int, default=0) #  index of backdoor in test transform
    parser.add_argument("--poisoned_target_transform_index", type=int, default=0) # index of backdoor in target transform 
    parser.add_argument("--log_iteration_interval", type=int, default=100)
    parser.add_argument("--test_epoch_interval", type=int, default=10)
    parser.add_argument("--save_epoch_interval", type=int, default=10)

    # BadNet attack
    parser.add_argument("--patch_size", type=int, default=3)

    # WaNet attack
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--grid_rescale", type=int, default=1)
    parser.add_argument("--noise_rescale", type=int, default=2)

    # FIBA attack https://github.com/HazardFY/FIBA
    parser.add_argument("--L", type=float, default=0.1)
    parser.add_argument("--mix_ratio", type=float, default=0.15)
    parser.add_argument("--target_img", type=str, default="/media/userdisk1/yf/input_aware_attack/coco_val75/000000002157.jpg")
    parser.add_argument("--cross_dir", type=str, default="/media/userdisk1/yf/input_aware_attack/coco_test1000")
    parser.add_argument("--cross_ratio", type=float, default=0)  # num_cross = int(num_bd * opt.cross_ratio)

    # CNP defense
    parser.add_argument('--nb_iter', type=int, default=4000, help='the number of iterations for training')# 默认是2000
    parser.add_argument('--print_every', type=int, default=500, help='print results every few iterations')
    parser.add_argument('--val_frac', type=float, default=0.02, help='The fraction of the validate set')
    parser.add_argument('--anp_eps', type=float, default=0.4)
    parser.add_argument('--anp_steps', type=int, default=1)
    parser.add_argument('--anp_alpha', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='./save')
    
    parser.add_argument('--mask_file', type=str, default='29mask_values.txt', help='The text file containing the mask values')
    parser.add_argument('--pruning_by', type=str, default='number', choices=['number', 'threshold'])
    parser.add_argument('--pruning_max', type=float, default=0.05, help='the maximum fraction for pruning')
    parser.add_argument('--pruning_step', type=float, default=0.0025, help='the step size for evaluating the pruning')

    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=66)

    # delete original target class in test set or not
    parser.add_argument("--include_ori_targetclass",type=bool,default=False)
    
    # lamda in loss
    parser.add_argument("--lamda",type=float,default=0.01)

    return parser
