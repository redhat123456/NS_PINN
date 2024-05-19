"""
PINN+incompressible NS equation
2-dimensional unsteady
PINN model +LOSS function
PINN融合不可压缩NS方程
二维非定常流动
PINN模型 + LOSS函数
"""
import os
import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs
from utils import gradient


class PINN_Net(nn.Module):
    def __init__(self, layer_mat, mean_value, std_value, device):
        super(PINN_Net, self).__init__()
        self.loss = nn.MSELoss()
        self.X_mean = torch.from_numpy(mean_value.astype(np.float32)).to(device)
        self.X_std = torch.from_numpy(std_value.astype(np.float32)).to(device)
        self.device = device
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            self.base.add_module(str(i) + "Act", nn.Tanh())
            # self.base.add_module(str(i) + "Drop", nn.Dropout(p=0.2))
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.Initial_param()

    def forward(self, X):
        # X = torch.cat(args, 1).requires_grad_(True).to(self.device)
        X_norm = (X - self.X_mean) / self.X_std
        predict = self.base(X_norm)
        return predict

    # 对参数进行初始化
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('linear.weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('linear.bias'):
                nn.init.zeros_(param)


class pinn_2d(PINN_Net):
    def NS_loss(self, inp, pre, Re):
        u = pre[:, 0:1]
        v = pre[:, 1:2]
        p = pre[:, 2:3]

        u_a = gradient(u, inp)
        v_a = gradient(v, inp)
        p_a = gradient(p, inp)

        u_x, u_y, u_t = u_a[:, 0:1], u_a[:, 1:2], u_a[:, 2:3]
        v_x, v_y, v_t = v_a[:, 0:1], v_a[:, 1:2], v_a[:, 2:3]
        p_x, p_y = p_a[:, 0:1], p_a[:, 1:2]

        u_xx = gradient(u_x, inp)[:, 0:1]
        u_yy = gradient(u_y, inp)[:, 1:2]
        v_xx = gradient(v_x, inp)[:, 0:1]
        v_yy = gradient(v_y, inp)[:, 1:2]

        # u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        # u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        # v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        # v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        # v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        # p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        # p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # # second-order derivative
        # # 二阶导
        # u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        # u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        # v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        # v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]

        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)

        # batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        f_t = torch.concat([f_equation_mass, f_equation_x, f_equation_y], 1)
        mse_NS = (f_t ** 2).mean()
        return mse_NS
