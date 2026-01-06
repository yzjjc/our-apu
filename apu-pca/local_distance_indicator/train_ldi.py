# import os
# import torch
# import sys
# sys.path.append(os.getcwd())
# import time
# from datetime import datetime
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from dataset.dataset import PUDataset
# from models.P2PNet_Attention import P2PNet
# from args.pu1k_args import parse_pu1k_args
# from args.pugan_args import parse_pugan_args
# from models.utils import *
# import argparse
# import math

# def update_learning_rate(iter_step, warm_up_end, max_iter, init_lr, optimizer):
#     warn_up = warm_up_end
#     lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
#     lr = lr * init_lr
#     for g in optimizer.param_groups:
#         g['lr'] = lr

# def train(args, exp_name):
#     set_seed(args.seed)
#     start = time.time()

#     # load data
#     train_dataset = PUDataset(args)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                    shuffle=True,
#                                                    batch_size=args.batch_size,
#                                                    num_workers=args.num_workers)
#     total_iter = args.epochs * len(train_loader)
#     # set up folders for checkpoints and logs
#     str_time = exp_name+'_'+datetime.now().isoformat()
#     output_dir = os.path.join(args.out_path, str_time)
#     ckpt_dir = os.path.join(output_dir, 'ckpt')
#     if not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     log_dir = os.path.join(output_dir, 'log')
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     writer = SummaryWriter(log_dir)
#     logger = get_logger('train', log_dir)
#     logger.info('Experiment ID: %s' % (str_time))

#     # create model
#     logger.info('========== Build Model ==========')
#     model = P2PNet(args)
#     model = model.cuda()
#     # get the parameter size
#     para_num = sum([p.numel() for p in model.parameters()])
#     logger.info("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))
#     # log
#     logger.info(args)
#     logger.info(repr(model))
#     # set model state
#     model.train()

#     # optimizer
#     assert args.optim in ['adam', 'sgd']
#     if args.optim == 'adam':
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         optimizer = optim.SGD(model.parameters(), lr=args.lr)
#     # lr scheduler
#     if args.scheduler_type == 'step':
#         scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

#     # train
#     logger.info('========== Begin Training ==========')
#     print("scheduler type:", args.scheduler_type)
#     iter_step = 0
#     for epoch in range(args.epochs):
#         logger.info('********* Epoch %d *********' % (epoch + 1))
#         # epoch loss
#         epoch_loss = 0.0

#         for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
#             iter_step += 1
#             if args.scheduler_type == 'cosine':
#                 update_learning_rate(iter_step, args.warm_up_end, total_iter, args.lr, optimizer)
#             # (b, n, 3) -> (b, 3, n)
#             input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
#             gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()

#             # midpoint interpolation
#             # interpolate_pts = input_pts
#             interpolate_pts = midpoint_interpolate(args, input_pts)

#             # query points
#             query_pts = get_query_points(interpolate_pts, args)
#             # model forward, predict point-to-point distance: (b, 1, n)
#             pred_p2p = model(interpolate_pts, query_pts)
#             # calculate loss
#             loss = get_p2p_loss(args, pred_p2p, query_pts, gt_pts)
#             epoch_loss += loss.item()

#             # update parameters
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # log
            
#             if (i+1) % args.print_rate == 0:
#                 logger.info("epoch: %d/%d, iters: %d/%d, lr: %f, loss: %f" %
#                       (epoch + 1, args.epochs, i + 1, len(train_loader), optimizer.param_groups[0]['lr'], epoch_loss / (i+1)))
#         writer.add_scalar('train/loss', epoch_loss / len(train_loader), epoch)
#         writer.flush()
#         # lr scheduler
#         if args.scheduler_type == 'step':
#             scheduler_steplr.step()

#         # log
#         interval = time.time() - start
#         logger.info("epoch: %d/%d, avg epoch loss: %f, time: %d mins %.1f secs" %
#           (epoch + 1, args.epochs, epoch_loss / len(train_loader), interval / 60, interval % 60))

#         # save checkpoint
#         if (epoch + 1) % args.save_rate == 0:
#             model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
#             model_path = os.path.join(ckpt_dir, model_name)
#             torch.save(model.state_dict(), model_path)
# import os
# import torch
# import torch.nn.functional as F
# import sys
# sys.path.append(os.getcwd())
# import time
# from datetime import datetime
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from args.utils import str2bool
# from dataset.dataset import PUDataset
# from models.P2PNet_Attention import P2PNet
# from args.pu1k_args import parse_pu1k_args
# from args.pugan_args import parse_pugan_args
# from models.utils import *
# import argparse
# import math


# def update_learning_rate(iter_step, warm_up_end, max_iter, init_lr, optimizer):
#     warn_up = warm_up_end
#     lr = (iter_step / warn_up) if iter_step < warn_up else \
#          0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
#     lr = lr * init_lr
#     for g in optimizer.param_groups:
#         g['lr'] = lr


# def compute_normal_consistency_loss(pred_normals, query_pts, gt_pts):
#     """
#     ✨ 新增: 计算法线一致性损失
#     Args:
#         pred_normals: (B, 3, M) - 预测的法线
#         query_pts: (B, 3, M) - 查询点
#         gt_pts: (B, 3, N) - GT 点云
#     Returns:
#         loss: scalar
#     """
#     B, _, M = query_pts.shape
    
#     # 1. 找到每个查询点的最近邻 GT 点
#     knn_pts = get_knn_pts(1, gt_pts, query_pts).squeeze(-1)  # (B, 3, M)
    
#     # 2. 计算向量: query -> nearest GT
#     vec_to_surface = knn_pts - query_pts  # (B, 3, M)
#     vec_to_surface = F.normalize(vec_to_surface, dim=1)
    
#     # 3. 法线应该与这个向量对齐 (余弦相似度)
#     cosine_sim = (pred_normals * vec_to_surface).sum(dim=1)  # (B, M)
    
#     # 4. 损失: 1 - |cos|  (希望 cos 接近 ±1)
#     loss = (1 - torch.abs(cosine_sim)).mean()
    
#     return loss


# def train(args, exp_name):
#     set_seed(args.seed)
#     start = time.time()

#     # 加载数据
#     train_dataset = PUDataset(args)
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         shuffle=True,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers
#     )
#     total_iter = args.epochs * len(train_loader)
    
#     # 设置输出目录
#     str_time = exp_name + '_' + datetime.now().isoformat()
#     output_dir = os.path.join(args.out_path, str_time)
#     ckpt_dir = os.path.join(output_dir, 'ckpt')
#     if not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     log_dir = os.path.join(output_dir, 'log')
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     writer = SummaryWriter(log_dir)
#     logger = get_logger('train', log_dir)
#     logger.info('Experiment ID: %s' % (str_time))

#     # 创建模型
#     logger.info('========== Build Model ==========')
#     model = P2PNet(args)
#     model = model.cuda()
    
#     para_num = sum([p.numel() for p in model.parameters()])
#     logger.info("=== Parameters: {:.4f} K === ".format(float(para_num / 1e3)))
#     logger.info(args)
#     logger.info(repr(model))
    
#     model.train()

#     # 优化器
#     assert args.optim in ['adam', 'sgd']
#     if args.optim == 'adam':
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
#     # 学习率调度器
#     if args.scheduler_type == 'step':
#         scheduler_steplr = optim.lr_scheduler.StepLR(
#             optimizer, step_size=args.lr_decay_step, gamma=args.gamma
#         )

#     # 训练
#     logger.info('========== Begin Training ==========')
#     logger.info("Scheduler type: %s" % args.scheduler_type)
#     logger.info("Use normal estimation: %s" % args.use_normal_estimation)
    
#     iter_step = 0
#     for epoch in range(args.epochs):
#         logger.info('********* Epoch %d *********' % (epoch + 1))
#         epoch_loss = 0.0
#         epoch_normal_loss = 0.0

#         for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
#             iter_step += 1
#             if args.scheduler_type == 'cosine':
#                 update_learning_rate(iter_step, args.warm_up_end, total_iter, args.lr, optimizer)
            
#             # 转换格式
#             input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
#             gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()

#             # 中点插值
#             interpolate_pts = midpoint_interpolate(args, input_pts)

#             # 查询点
#             query_pts = get_query_points(interpolate_pts, args)
            
#             # ✨ 修改: 前向传播 (可能返回法线)
#             if args.use_normal_estimation:
#                 pred_p2p, pred_normals = model(
#                     interpolate_pts, query_pts, 
#                     return_normals=True
#                 )
#             else:
#                 pred_p2p = model(interpolate_pts, query_pts)
            
#             # 计算距离损失
#             loss_p2p = get_p2p_loss(args, pred_p2p, query_pts, gt_pts)
            
#             # ✨ 计算法线损失 (如果启用)
#             loss_normal = torch.tensor(0.0).cuda()
#             if args.use_normal_estimation:
#                 loss_normal = compute_normal_consistency_loss(
#                     pred_normals, query_pts, gt_pts
#                 )
            
#             # 总损失
#             loss = loss_p2p + args.normal_loss_weight * loss_normal
            
#             epoch_loss += loss_p2p.item()
#             epoch_normal_loss += loss_normal.item()

#             # 更新参数
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # 日志
#             if (i+1) % args.print_rate == 0:
#                 logger.info(
#                     "epoch: %d/%d, iters: %d/%d, lr: %f, "
#                     "loss_p2p: %.6f, loss_normal: %.6f, total: %.6f" %
#                     (epoch + 1, args.epochs, i + 1, len(train_loader), 
#                      optimizer.param_groups[0]['lr'], 
#                      loss_p2p.item(), loss_normal.item(), loss.item())
#                 )
        
#         # TensorBoard 记录
#         writer.add_scalar('train/loss_p2p', epoch_loss / len(train_loader), epoch)
#         if args.use_normal_estimation:
#             writer.add_scalar('train/loss_normal', epoch_normal_loss / len(train_loader), epoch)
#         writer.flush()
        
#         # 学习率调度
#         if args.scheduler_type == 'step':
#             scheduler_steplr.step()

#         # 日志
#         interval = time.time() - start
#         logger.info(
#             "epoch: %d/%d, avg loss_p2p: %.6f, avg loss_normal: %.6f, time: %d mins %.1f secs" %
#             (epoch + 1, args.epochs, 
#              epoch_loss / len(train_loader),
#              epoch_normal_loss / len(train_loader),
#              interval / 60, interval % 60)
#         )

#         # 保存模型
#         if (epoch + 1) % args.save_rate == 0:
#             model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
#             model_path = os.path.join(ckpt_dir, model_name)
#             torch.save(model.state_dict(), model_path)

# # def parse_train_args():
# #     parser = argparse.ArgumentParser(description='Training Arguments')
# #     parser.add_argument('--exp_name', default='exp', type=str)
# #     parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
# #     parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
# #     parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
# #     parser.add_argument('--epochs', default=60, type=int, help='training epochs')
# #     parser.add_argument('--batch_size', default=32, type=int, help='batch size')
# #     parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
# #     parser.add_argument('--save_rate', default=10, type=int, help='model save frequency')
# #     parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')
# #     parser.add_argument('--scheduler_type', default='step', type=str, help='step or cosine; type of learning rate scheduler')
    
# #     args = parser.parse_args()
# #     return args
# def parse_train_args():
#     """只解析训练特有的参数（不包括模型参数）"""
#     parser = argparse.ArgumentParser(description='Training Arguments')
    
#     # 基本参数
#     parser.add_argument('--exp_name', default='exp', type=str)
#     parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    
#     # 训练参数（可能覆盖模型默认值的）
#     parser.add_argument('--epochs', type=int, help='training epochs')
#     parser.add_argument('--batch_size', type=int, help='batch size')
#     parser.add_argument('--lr', type=float, help='learning rate')
#     parser.add_argument('--scheduler_type', type=str, help='step or cosine')
#     parser.add_argument('--print_rate', type=int, help='print frequency')
#     parser.add_argument('--save_rate', type=int, help='save frequency')
    
#     # ✨ 法线估计参数（可选覆盖）
#     parser.add_argument('--use_normal_estimation', type=str2bool, help='use normal estimation')
#     parser.add_argument('--normal_loss_weight', type=float, help='normal loss weight')
#     parser.add_argument('--normal_estimation_method', type=str, help='pca or learned')
#     parser.add_argument('--k_normal_estimation', type=int, help='k for PCA normal')
    
#     # ✨ 使用 parse_known_args 而不是 parse_args
#     # 这样可以忽略未知参数（来自模型配置的参数）
#     args, unknown = parser.parse_known_args()
    
#     return args

# # if __name__ == "__main__":
# #     train_args = parse_train_args()
# #     exp_name = train_args.exp_name
# #     assert train_args.dataset in ['pu1k', 'pugan']
# #     if train_args.dataset == 'pu1k':
# #         model_args = parse_pu1k_args()
# #     else:
# #         model_args = parse_pugan_args()
# #     reset_model_args(train_args, model_args)

# #     train(model_args, exp_name)
# if __name__ == "__main__":
#     train_args = parse_train_args()
#     assert train_args.dataset in ['pu1k', 'pugan']
    
#     if train_args.dataset == 'pu1k':
#         model_args = parse_pu1k_args()
#     else:
#         model_args = parse_pugan_args()
    
#     reset_model_args(train_args, model_args)
#     train(model_args, train_args.exp_name)

import os
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.getcwd())
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import PUDataset
from models.P2PNet_Attention import P2PNet
from args.pu1k_args import parse_pu1k_args
from args.pugan_args import parse_pugan_args
from models.utils import *
from args.utils import str2bool
import argparse
import math


def update_learning_rate(iter_step, warm_up_end, max_iter, init_lr, optimizer):
    warn_up = warm_up_end
    lr = (iter_step / warn_up) if iter_step < warn_up else \
         0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
    lr = lr * init_lr
    for g in optimizer.param_groups:
        g['lr'] = lr


def compute_normal_consistency_loss(pred_normals, query_pts, gt_pts):
    """计算法线一致性损失"""
    B, _, M = query_pts.shape
    
    knn_pts = get_knn_pts(1, gt_pts, query_pts).squeeze(-1)
    vec_to_surface = knn_pts - query_pts
    vec_to_surface = F.normalize(vec_to_surface, dim=1)
    cosine_sim = (pred_normals * vec_to_surface).sum(dim=1)
    loss = (1 - torch.abs(cosine_sim)).mean()
    
    return loss


def train(args, exp_name):
    set_seed(args.seed)
    start = time.time()

    train_dataset = PUDataset(args)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    total_iter = args.epochs * len(train_loader)
    
    str_time = exp_name + '_' + datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger('train', log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    logger.info('========== Build Model ==========')
    model = P2PNet(args)
    model = model.cuda()
    
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info("=== Parameters: {:.4f} K === ".format(float(para_num / 1e3)))
    logger.info("Use normal estimation: %s" % args.use_normal_estimation)
    if args.use_normal_estimation:
        logger.info("Normal estimation method: %s" % args.normal_estimation_method)
        if args.normal_estimation_method == 'pca':
            logger.info("K neighbors for normal: %d" % args.k_normal_estimation)
        logger.info("Normal loss weight: %.4f" % args.normal_loss_weight)
    
    model.train()

    assert args.optim in ['adam', 'sgd']
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    if args.scheduler_type == 'step':
        scheduler_steplr = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.gamma
        )

    logger.info('========== Begin Training ==========')
    
    iter_step = 0
    for epoch in range(args.epochs):
        logger.info('********* Epoch %d *********' % (epoch + 1))
        epoch_loss = 0.0
        epoch_loss_p2p = 0.0
        epoch_loss_normal = 0.0

        for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
            iter_step += 1
            if args.scheduler_type == 'cosine':
                update_learning_rate(iter_step, args.warm_up_end, total_iter, args.lr, optimizer)
            
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()

            interpolate_pts = midpoint_interpolate(args, input_pts)
            query_pts = get_query_points(interpolate_pts, args)
            
            if args.use_normal_estimation:
                pred_p2p, pred_normals = model(
                    interpolate_pts, query_pts, 
                    return_normals=True
                )
            else:
                pred_p2p = model(interpolate_pts, query_pts)
            
            loss_p2p = get_p2p_loss(args, pred_p2p, query_pts, gt_pts)
            
            loss_normal = torch.tensor(0.0).cuda()
            if args.use_normal_estimation:
                loss_normal = compute_normal_consistency_loss(
                    pred_normals, query_pts, gt_pts
                )
            
            loss = loss_p2p + args.normal_loss_weight * loss_normal
            
            epoch_loss += loss.item()
            epoch_loss_p2p += loss_p2p.item()
            epoch_loss_normal += loss_normal.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % args.print_rate == 0:
                logger.info(
                    "epoch: %d/%d, iters: %d/%d, lr: %.6f, "
                    "loss_p2p: %.6f, loss_normal: %.6f, total: %.6f" %
                    (epoch + 1, args.epochs, i + 1, len(train_loader), 
                     optimizer.param_groups[0]['lr'], 
                     loss_p2p.item(), loss_normal.item(), loss.item())
                )
        
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_p2p = epoch_loss_p2p / len(train_loader)
        avg_loss_normal = epoch_loss_normal / len(train_loader)
        
        writer.add_scalar('train/loss_total', avg_loss, epoch)
        writer.add_scalar('train/loss_p2p', avg_loss_p2p, epoch)
        if args.use_normal_estimation:
            writer.add_scalar('train/loss_normal', avg_loss_normal, epoch)
        writer.flush()
        
        if args.scheduler_type == 'step':
            scheduler_steplr.step()

        interval = time.time() - start
        logger.info(
            "epoch: %d/%d, avg_loss: %.6f, avg_p2p: %.6f, avg_normal: %.6f, "
            "time: %d mins %.1f secs" %
            (epoch + 1, args.epochs, avg_loss, avg_loss_p2p, avg_loss_normal,
             interval / 60, interval % 60)
        )

        if (epoch + 1) % args.save_rate == 0:
            model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
            model_path = os.path.join(ckpt_dir, model_name)
            torch.save(model.state_dict(), model_path)
            logger.info("Model saved: %s" % model_path)


def create_args_from_cmd():
    """
    从命令行创建完整的参数配置
    """
    # 1. 创建临时解析器获取数据集类型
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--dataset', default='pugan', type=str)
    temp_args, _ = temp_parser.parse_known_args()
    
    # 2. 根据数据集获取默认配置
    if temp_args.dataset == 'pu1k':
        args = parse_pu1k_args()
    else:
        args = parse_pugan_args()
    
    # 3. 创建完整解析器（只包含可能覆盖的参数）
    parser = argparse.ArgumentParser(description='Training Arguments')
    
    # 实验参数
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--dataset', type=str, default='pugan')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--print_rate', type=int, default=None)
    parser.add_argument('--save_rate', type=int, default=None)
    
    # 法线估计参数
    parser.add_argument('--use_normal_estimation', type=str2bool, default=None)
    parser.add_argument('--normal_loss_weight', type=float, default=None)
    parser.add_argument('--normal_estimation_method', type=str, default=None)
    parser.add_argument('--k_normal_estimation', type=int, default=None)
    
    # 4. 解析命令行参数
    cmd_args = parser.parse_args()
    
    # 5. 用命令行参数覆盖默认配置（只覆盖非None的值）
    for key, value in vars(cmd_args).items():
        if value is not None:
            setattr(args, key, value)
    
    return args, cmd_args.exp_name


if __name__ == "__main__":
    args, exp_name = create_args_from_cmd()
    train(args, exp_name)