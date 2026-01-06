import argparse
from args.utils import str2bool


def parse_pu1k_args():
    parser = argparse.ArgumentParser(description='Model Arguments')
    # seed
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--seed', default=21, type=float, help='seed')
    # optimizer
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    # lr scheduler
    parser.add_argument('--scheduler_type', default='step', type=str, help='step or cosine; type of learning rate scheduler')
    parser.add_argument('--warm_up_end', default=250, type=int, help='warm up end epoch of cosine scheduler')
    parser.add_argument('--lr_decay_step', default=20, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    # dataset
    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="./data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5", type=str, help='the path of train dataset')
    parser.add_argument('--num_points', default=256, type=int, help='the points number of each input patch')
    parser.add_argument('--skip_rate', default=1, type=int, help='used for dataset')
    parser.add_argument('--use_random_input', default=False, type=str2bool, help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
    # train
    parser.add_argument('--epochs', default=60, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers number')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=10, type=int, help='model save frequency')
    parser.add_argument('--use_smooth_loss', default=False, type=str2bool, help='whether use smooth L1 loss')
    parser.add_argument('--beta', default=0.01, type=float, help='beta for smooth L1 loss')
    # model
    parser.add_argument('--k', default=16, type=int, help='neighbor number')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--block_num', default=3, type=int, help='dense block number in the feature extractor')
    parser.add_argument('--layer_num', default=3, type=int, help='dense layer number in each dense block')
    parser.add_argument('--feat_dim', default=32, type=int, help='input(output) feature dimension in each dense block' )
    parser.add_argument('--bn_size', default=1, type=int, help='the factor used in the bottleneck layer')
    parser.add_argument('--growth_rate', default=32, type=int, help='output feature dimension in each dense layer')
    # query points
    parser.add_argument('--local_sigma', default=0.02, type=float, help='used for sample points')
    # truncate distance
    parser.add_argument('--truncate_distance', default=False, type=str2bool, help='whether truncate distance')
    parser.add_argument('--max_dist', default=0.2, type=float, help='the maximum point-to-point distance')
    # ouput
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')
    # test
    parser.add_argument('--num_iterations', default=10, type=int, help='the number of update iterations')
    parser.add_argument('--test_step_size', default=500, type=float, help='predefined test step size')
    parser.add_argument('--test_input_path', default='./data/PU1K/test/input_2048/input_2048/', type=str, help='the test input data path')
    parser.add_argument('--ckpt_path', default='./pretrained_model/pu1k/ckpt/ckpt-epoch-60.pth', type=str, help='the pretrained model path')
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud')
    parser.add_argument('--double_4X', default=False, type=str2bool, help='conduct 4X twice to get 16X')
    
    # global
    parser.add_argument('--conf', type=str, default='./confs/pugan.conf')
    parser.add_argument('--mode', type=str, default='train or upsample')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='')
    parser.add_argument('--listname', type=str, default='pugan.txt')

    args = parser.parse_args()

    return args


def parse_pu1k_args_multiscale():
    """
    Multi-scale version of PU1K arguments
    Adds multi-scale feature extraction parameters
    """
    parser = argparse.ArgumentParser(description='Model Arguments - Multi-Scale PU1K')
    
    # ========== Basic Settings ==========
    parser.add_argument('--exp_name', default='exp_multiscale_pu1k', type=str,
                       help='experiment name')
    parser.add_argument('--seed', default=21, type=float, help='random seed')
    
    # ========== Optimizer Settings ==========
    parser.add_argument('--optim', default='adam', type=str, 
                       help='optimizer type: adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, 
                       help='initial learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, 
                       help='weight decay for optimizer')
    
    # ========== Learning Rate Scheduler ==========
    parser.add_argument('--scheduler_type', default='step', type=str, 
                       choices=['step', 'cosine'],
                       help='learning rate scheduler type')
    parser.add_argument('--warm_up_end', default=250, type=int, 
                       help='warm up end iteration for cosine scheduler')
    parser.add_argument('--lr_decay_step', default=20, type=int, 
                       help='learning rate decay step size for step scheduler')
    parser.add_argument('--gamma', default=0.5, type=float, 
                       help='learning rate decay factor for step scheduler')
    
    # ========== Dataset Settings ==========
    parser.add_argument('--dataset', default='pu1k', type=str, 
                       help='dataset name: pu1k or pugan')
    parser.add_argument('--h5_file_path', 
                       default="./data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5", 
                       type=str, help='path to training dataset (HDF5 file)')
    parser.add_argument('--num_points', default=256, type=int, 
                       help='number of points in each input patch')
    parser.add_argument('--skip_rate', default=1, type=int, 
                       help='skip rate for dataset loading')
    parser.add_argument('--use_random_input', default=False, type=str2bool, 
                       help='whether to use random sampling for input generation')
    
    # ========== Data Augmentation ==========
    parser.add_argument('--jitter_sigma', type=float, default=0.01, 
                       help='standard deviation for jitter augmentation')
    parser.add_argument('--jitter_max', type=float, default=0.03, 
                       help='maximum jitter value (clipping threshold)')
    
    # ========== Training Settings ==========
    parser.add_argument('--epochs', default=60, type=int, 
                       help='total number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, 
                       help='batch size for training')
    parser.add_argument('--num_workers', default=4, type=int, 
                       help='number of data loading workers')
    parser.add_argument('--print_rate', default=200, type=int, 
                       help='frequency (in iterations) to print training loss')
    parser.add_argument('--save_rate', default=10, type=int, 
                       help='frequency (in epochs) to save model checkpoint')
    
    # ========== Loss Settings ==========
    parser.add_argument('--use_smooth_loss', default=False, type=str2bool, 
                       help='whether to use smooth L1 loss instead of L1 loss')
    parser.add_argument('--beta', default=0.01, type=float, 
                       help='beta parameter for smooth L1 loss')
    
    # ========== Model Architecture (Original) ==========
    parser.add_argument('--k', default=16, type=int, 
                       help='number of neighbors for k-NN (base scale, may not be used in multi-scale)')
    parser.add_argument('--up_rate', default=4, type=int, 
                       help='upsampling rate (e.g., 4 for 4x upsampling)')
    parser.add_argument('--block_num', default=3, type=int, 
                       help='number of dense blocks in feature extractor')
    parser.add_argument('--layer_num', default=3, type=int, 
                       help='number of dense layers in each dense block')
    parser.add_argument('--feat_dim', default=32, type=int, 
                       help='input/output feature dimension for each dense block')
    parser.add_argument('--bn_size', default=1, type=int, 
                       help='bottleneck size factor (multiplier for growth_rate)')
    parser.add_argument('--growth_rate', default=32, type=int, 
                       help='growth rate (output feature dimension in each dense layer)')
    
    # ========== Multi-Scale Parameters (NEW) ==========
    parser.add_argument('--scales', nargs='+', type=int, default=[8, 16, 32],
                       help='Multi-scale k values for k-NN feature extraction. '
                            'Example: [8, 16, 32] means extracting features at 3 scales. '
                            'Smaller k captures fine details, larger k captures global structure.')
    
    parser.add_argument('--fusion_type', default='hierarchical', type=str,
                       choices=['concat', 'attention', 'hierarchical'],
                       help='Type of multi-scale feature fusion strategy:\n'
                            '  - concat: simple concatenation followed by MLP fusion\n'
                            '  - attention: attention-weighted fusion across scales\n'
                            '  - hierarchical: hierarchical fusion from fine to coarse (recommended)')
    
    parser.add_argument('--use_multiscale', default=True, type=str2bool,
                       help='Whether to use multi-scale feature extraction. '
                            'Set to False to use original single-scale model.')
    
    # ========== Query Points Sampling ==========
    parser.add_argument('--local_sigma', default=0.02, type=float, 
                       help='standard deviation for sampling query points around surface')
    
    # ========== Distance Truncation ==========
    parser.add_argument('--truncate_distance', default=False, type=str2bool, 
                       help='whether to truncate predicted distances to a maximum value')
    parser.add_argument('--max_dist', default=0.2, type=float, 
                       help='maximum distance value for truncation')
    
    # ========== Output Settings ==========
    parser.add_argument('--out_path', default='./output', type=str, 
                       help='output directory for checkpoints and logs')
    
    # ========== Test Settings ==========
    parser.add_argument('--num_iterations', default=10, type=int, 
                       help='number of iterative update iterations during testing')
    parser.add_argument('--test_step_size', default=500, type=float, 
                       help='step size for gradient-based point cloud update during testing')
    parser.add_argument('--test_input_path', 
                       default='./data/PU1K/test/input_2048/input_2048/', 
                       type=str, help='path to test input point clouds')
    parser.add_argument('--ckpt_path', 
                       default='./pretrained_model/pu1k_multiscale/ckpt/ckpt-epoch-60.pth', 
                       type=str, help='path to pretrained model checkpoint')
    parser.add_argument('--patch_rate', default=3, type=int, 
                       help='patch extraction rate for testing (controls number of patches)')
    parser.add_argument('--save_dir', default='pcd', type=str, 
                       help='directory name for saving upsampled point clouds')
    parser.add_argument('--double_4X', default=False, type=str2bool, 
                       help='whether to conduct 4X upsampling twice to achieve 16X upsampling')
    
    # ========== Global Field Settings ==========
    parser.add_argument('--conf', type=str, default='./confs/pu1k.conf',
                       help='configuration file for global field training')
    parser.add_argument('--mode', type=str, default='train',
                       help='running mode: train or upsample')
    parser.add_argument('--dir', type=str, default='',
                       help='experiment directory name for global field')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--dataname', type=str, default='',
                       help='specific shape name for global field processing')
    parser.add_argument('--listname', type=str, default='pu1k.txt',
                       help='file containing list of shapes for batch processing')

    args = parser.parse_args()
    
    # # ========== Argument Validation ==========
    # # Validate scales
    # if args.use_multiscale:
    #     if len(args.scales) < 2:
    #         parser.error("At least 2 scales are required for multi-scale feature extraction")
    #     if any(k <= 0 for k in args.scales):
    #         parser.error("All scale values must be positive integers")
    #     if args.scales != sorted(args.scales):
    #         print(f"Warning: Scales {args.scales} are not sorted. Sorting to {sorted(args.scales)}")
    #         args.scales = sorted(args.scales)
    
    # # Validate fusion type
    # if args.use_multiscale and args.fusion_type not in ['concat', 'attention', 'hierarchical']:
    #     parser.error(f"Invalid fusion_type: {args.fusion_type}. Must be one of [concat, attention, hierarchical]")
    
    # # Print multi-scale configuration
    # if args.use_multiscale:
    #     print("=" * 60)
    #     print("Multi-Scale Configuration:")
    #     print(f"  Scales (k values): {args.scales}")
    #     print(f"  Fusion strategy: {args.fusion_type}")
    #     print(f"  Number of scales: {len(args.scales)}")
    #     print("=" * 60)
    
    return args
# 在两个文件中都添加以下参数

def parse_pugan_args():  # 或 parse_pu1k_args()
    parser = argparse.ArgumentParser(prog="pugan")
    
    # ... 现有参数 ...
    
    # ✨ 新增: 法线估计相关参数
    parser.add_argument('--use_normal_estimation', default=True, type=str2bool, 
                       help='whether to use normal estimation module')
    parser.add_argument('--normal_loss_weight', default=0.1, type=float,
                       help='weight for normal consistency loss')
    
    args = parser.parse_args()
    return args