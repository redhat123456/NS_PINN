# coding:utf-8
import collections
import traceback
import torch
from torch import nn

from model import PINN_2D, NS_loss
from utils import jy_deal, add_log, train_bar, write_log, save_csv, ini_env, build_optimizer, validation_2d, load_data, \
    select_point, load_equation_points_lhs
import argparse
import os
import time
import numpy as np
import pynvml

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
# parser.add_argument('--name', type=str, default='Re3900_2024-05-20 09h 52m 54s')
parser.add_argument('--data_path', type=str, default='data/data_Re3900_2.mat')
parser.add_argument('--Re', type=int, default=3900)
parser.add_argument('--L', type=float, default=0.1)
parser.add_argument('--U', type=float, default=1.0)
parser.add_argument('--left', type=float, default=-0.8)
parser.add_argument('--right', type=float, default=1.5)
parser.add_argument('--bottom', type=float, default=-1.0)
parser.add_argument('--top', type=float, default=1.0)
parser.add_argument('--r', type=float, default=0.05)
parser.add_argument('--epoch', type=int, default=500000)
parser.add_argument('--inner_epoch', type=int, default=1)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--dimension', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_layers', type=int, default=10)
parser.add_argument('--layer_neurons', type=int, default=64)
parser.add_argument('--N_data', type=int, default=100000)
parser.add_argument('--N_eqa', type=int, default=50000)
parser.add_argument('--N_bound', type=int, default=10000)
parser.add_argument('--is_print', type=bool, default=True)
args = parser.parse_args()


# left = -0.05
# right = 0.2
# bottom = -0.05
# top = 0.05
# center_point = [0, 0]
# r = 0.005


class TrainModel(object):
    def __init__(self, args, data_path, Re, L, U):
        self.device, self.gpus, self.device_name = ini_env()
        fileName = os.path.basename(data_path)
        dirStr, ext = os.path.splitext(fileName)
        self.model_name = dirStr + '_2d'
        self.data_path = data_path
        self.Re = Re
        self.L = L
        self.U = U
        self.lr = args.lr
        self.epoch = args.epoch
        self.inner_epoch = args.inner_epoch
        self.layer_mat = [args.dimension] + args.hidden_layers * [args.layer_neurons] + [3]
        self.dimension = args.dimension
        self.N_eqa = args.N_eqa
        self.N_bound = args.N_bound
        self.N_data = args.N_data
        self.save_interval = args.save_interval
        self.xy_range = [args.left, args.right, args.bottom, args.top]
        self.r = args.r
        self.name = args.name

    def prepare_training(self):
        txt_list = []
        data, total_data, total_data_no_state, bound_data, data_mean, data_std = load_data(self.data_path, self.L,
                                                                                           self.U)
        self.data_mean = data_mean
        self.data_std = data_std
        self.inp_mean = data_mean[:3]
        self.inp_std = data_std[:3]
        self.out_mean = data_mean[3:]
        self.out_std = data_std[3:]

        if self.name is None:
            re_des = f"Re{self.Re}_"
            self.name = re_des + time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
            self.write_path = os.path.join('log', self.name)
            train_cfg = {
                'name': self.name,
                'model_name': self.model_name,
                'write_path': self.write_path,
                'data_path': self.data_path,
                'Re': self.Re,
                'L': self.L,
                'U': self.U,
                'layer_mat': self.layer_mat,
                'epoch': self.epoch,
                'inner_epoch': self.inner_epoch,
                'lr': self.lr,
                'dimension': self.dimension,
                'N_eqa': self.N_eqa,
                'N_bound': self.N_bound,
                'N_data': self.N_data,
                'save_interval': self.save_interval,
                'xy_range': self.xy_range,
                'r': self.r,
                'data_mean': self.data_mean.tolist(),
                'data_std': self.data_std.tolist(),
            }
            jy_deal(os.path.join(self.write_path, 'train_cfg.yaml'), 'w', train_cfg)
            os.makedirs(self.write_path, exist_ok=True)
            self.is_continue = False
        else:
            self.write_path = os.path.join('log', self.name)
            if not os.path.exists(self.write_path):
                raise FileNotFoundError(f'{self.name} not found')
            else:
                train_cfg = jy_deal(os.path.join(self.write_path, 'train_cfg.yaml'), 'r')
                self.name = train_cfg['name']
                self.model_name = train_cfg['model_name']
                self.data_path = train_cfg['data_path']
                self.Re = train_cfg['Re']
                self.L = train_cfg['L']
                self.U = train_cfg['U']
                self.layer_mat = train_cfg['layer_mat']
                self.epoch = train_cfg['epoch']
                self.inner_epoch = train_cfg['inner_epoch']
                self.lr = train_cfg['lr']
                self.dimension = train_cfg['dimension']
                self.N_eqa = train_cfg['N_eqa']
                self.N_bound = train_cfg['N_bound']
                self.N_data = train_cfg['N_data']
                self.save_interval = train_cfg['save_interval']
                self.xy_range = train_cfg['xy_range']
                self.r = train_cfg['r']
                self.data_mean = np.array(train_cfg['data_mean'])
                self.data_std = np.array(train_cfg['data_std'])
                self.inp_mean = data_mean[:3]
                self.inp_std = data_std[:3]
                self.out_mean = data_mean[3:]
                self.out_std = data_std[3:]
                self.is_continue = True

        model = PINN_2D(self.layer_mat, self.inp_mean, self.inp_std, self.device)

        add_log('{0:-^60}'.format('训练准备中'), txt_list)
        add_log('{0:-^60}'.format('模型与训练参数如下'), txt_list)
        add_log(f'训练模型为:{self.model_name}', txt_list)
        add_log(
            f'训练迭代数为:{self.epoch},初始学习率为:{self.lr}', txt_list)
        add_log(f'监测数据量为:{len(data)}，采样数为{self.N_data}', txt_list)

        add_log(f'方程采样数为:{self.N_eqa}', txt_list)
        add_log(f'边界数据量为:{len(bound_data)}，采样数为{self.N_bound}', txt_list)
        add_log('计算平台: ', txt_list)
        for i in self.device_name:
            add_log(i, txt_list)

        return [model, txt_list, data, total_data, total_data_no_state, bound_data, self.data_mean, self.data_std]

    def process_training(self, train_args):
        model, txt_list, data, total_data, total_data_no_state, bound_data, data_mean, data_std = train_args
        pynvml.nvmlInit()
        last_model_file = f"{self.write_path}/last_{self.model_name}.pth"
        best_model_file = f"{self.write_path}/best_{self.model_name}.pth"
        loss_path = os.path.join(self.write_path, 'loss.csv')
        evaluate_path = os.path.join(self.write_path, 'evaluate.csv')

        add_log('{0:-^60}'.format('训练开始'), txt_list)

        if self.is_continue:
            model_comment = torch.load(last_model_file)
            if isinstance(model_comment, collections.OrderedDict):
                model.load_state_dict(
                    {k.replace('module.', ''): v for k, v in model_comment.items()})
            loss_data = np.loadtxt(f"{loss_path}", delimiter=',')
            t_list = loss_data[:, 0].tolist()
            lost_list = loss_data[:, 1].tolist()
            min_lost = min(lost_list)
            ml_epoch = np.argmin(lost_list) + 1
            start_epoch = len(lost_list) * self.save_interval
        else:
            t_list = []
            min_lost = np.Inf  # 起步loss
            ml_epoch = 1  # 标记最低损失的迭代数
            start_epoch = 0

        model = model.to(self.device)  # 将模型迁移到gpu
        model = nn.DataParallel(model, device_ids=self.gpus, output_device=self.gpus[0])
        loss_fn = nn.MSELoss()

        optimizer, scheduler = build_optimizer(model, self.lr)
        optimizer.zero_grad()  # 梯度归零
        for i in range(start_epoch):
            optimizer.step()
            scheduler.step()

        a = ("%-11s" + "%-15s" + "%-11s" + "%-14s" + "%-17s" + "%-33s" + "%-20s") % (
            "Epoch",
            "GPU0_men",
            "GPU0_use",
            "train_loss",
            "inner_process",
            "epoch_process",
            "time"
        )
        if args.is_print:
            print(a)

        # data = torch.tensor(data, dtype=torch.float32).to(self.device)

        date_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        st = time.perf_counter()
        ss_time = time.time()  # 开始时间

        min_t = total_data_no_state[:, 2].min()
        max_t = total_data_no_state[:, 2].max()
        low_bound = np.array([self.xy_range[0] / self.L, self.xy_range[2] / self.L, min_t]).reshape(1, -1)
        up_bound = np.array([self.xy_range[1] / self.L, self.xy_range[3] / self.L, max_t]).reshape(1, -1)
        radius = self.r / self.L
        loss_list = []

        for i in range(start_epoch, self.epoch):
            model.train()
            for j in range(self.inner_epoch):
                eqa_dataset = load_equation_points_lhs(low_bound, up_bound, self.dimension, self.N_eqa, radius)

                bound_dataset = select_point(bound_data, self.N_bound, is_dis=False).to(self.device)

                data_dataset = select_point(data, self.N_data, is_dis=False).to(self.device)

                self.single_train(model, optimizer, scheduler, data_dataset, eqa_dataset, bound_dataset, loss_list,
                                  loss_fn)

                if args.is_print:
                    log = [self.epoch, self.inner_epoch, st, i + 1, j + 1, loss_list[-1][0], start_epoch]
                    train_bar(log)

            if (i + 1) % self.save_interval == 0:
                train_loss = loss_list[-1][0]
                data_loss = loss_list[-1][1]
                eqa_loss = loss_list[-1][2]
                bound_loss = loss_list[-1][3]

                # sigma_data = float(sigma1.item())
                # sigma_ns = float(sigma2.item())
                # sigma_bound = float(sigma3.item())

                loss_list.append(train_loss)
                current_lr = optimizer.param_groups[0]['lr']

                valid_u, valid_v, valid_p = validation_2d(total_data, self.out_mean, self.out_std, model, self.L,
                                                          self.U,
                                                          self.device)
                if args.is_print:
                    print()
                    print(f"L2_u: {valid_u}")
                    print(f"L2_v: {valid_v}")
                    print(f"L2_p: {valid_p}")
                    print(a)

                torch.save(model.state_dict(), f'{last_model_file}')
                if train_loss < min_lost:
                    ml_epoch = i + 1
                min_lost = train_loss
                torch.save(model.state_dict(), f'{best_model_file}')
                ee_time = time.time()
                save_csv(loss_path, ee_time - ss_time, train_loss, data_loss, eqa_loss, bound_loss, current_lr)
                save_csv(evaluate_path, valid_u, valid_v, valid_p)

                t_list.append(ee_time - ss_time)
                ss_time = ee_time

        edate_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        total_time = sum(t_list)

        add_log('{0:-^60}'.format(f'本次训练结束于{edate_time}，训练结果如下'), txt_list)
        add_log(f'本次训练开始时间：{date_time}', txt_list)
        add_log(
            "本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                     int(total_time % 60)), txt_list)

        add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(float(min_lost), 3)}', txt_list)

        write_log(f"{self.write_path}", self.model_name, txt_list)

    def single_train(self, model, optimizer, scheduler, data_dataset, eqa_dataset, bound_dataset, loss_list, loss_fn):

        out_mean = torch.tensor(self.out_mean, dtype=torch.float32).to(self.device)

        out_std = torch.tensor(self.out_std, dtype=torch.float32).to(self.device)

        optimizer.zero_grad()  # 梯度归零

        # with autocast(): 混合精度训练

        eqa_inp = eqa_dataset.requires_grad_(True).to(self.device)

        n_bound = bound_dataset.shape[0]

        bd_dataset = torch.cat((data_dataset, bound_dataset), 0)

        bd_inp = bd_dataset[:, 0:3]

        bd_true = bd_dataset[:, 3:]

        bd_true = (bd_true - out_mean) / out_std  # 监督值标准化

        predict_bd = model(bd_inp)

        pre_eqa = model(eqa_inp)

        pre_eqa = pre_eqa * out_std + out_mean  # 方程预测值反标准化

        eqa_loss = NS_loss(eqa_inp, pre_eqa, self.Re)

        bound_loss = loss_fn(bd_true[:n_bound, :], predict_bd[:n_bound, :])

        data_loss = loss_fn(bd_true[n_bound:, :], predict_bd[n_bound:, :])

        loss = data_loss + eqa_loss + bound_loss

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 梯度优化
        scheduler.step()

        train_loss = float(loss.item())
        data_loss = float(data_loss.item())
        eqa_loss = float(eqa_loss.item())
        bound_loss = float(bound_loss.item())

        loss_list.append([train_loss, data_loss, eqa_loss, bound_loss])


def train(data_path, Re, L, U):
    model = TrainModel(args, data_path, Re, L, U)
    train_args = model.prepare_training()
    model.process_training(train_args)


if __name__ == '__main__':
    try:
        train(args.data_path, args.Re, args.L, args.U)
    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/error.log', 'w'))
