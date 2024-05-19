import argparse

import imageio
import numpy as np
import pandas as pd
import scipy
import matplotlib
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import *
from utils import jy_deal, no_state, load_data, validation_2d

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Re3900_2024-05-18 11h 39m 57s')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fan_wulianggang(u, v, p, U, rou):
    P = rou * U ** 2
    u_ = u * U
    v_ = v * U
    p_ = p * P
    return u_, v_, p_


def make_flow_gif(filename, time_series, name='q', fps_num=5):
    gif_images = []
    gif_name = os.path.join(filename, name + '.gif')
    img_name = os.path.join(filename, 'gif_make')
    for select_time in time_series:
        time = select_time.item()
        gif_images.append(imageio.v2.imread(img_name + '/--time' + "{:.2f}".format(time) + name + '.png'))
    imageio.mimsave(gif_name, gif_images, fps=fps_num, duration=0.1, loop=0)


def plot_compare_time_series(filename, x_mesh, y_mesh, q_selected, q_predict, select_time, v_norm, name='q'):
    plt.cla()
    # v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = ax1.contourf(x_mesh, y_mesh, q_selected, levels=200, cmap='jet', norm=v_norm)
    fig.colorbar(mappable, cax=cbar_ax)
    ax1.set_title("True_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    ax2.contourf(x_mesh, y_mesh, q_predict, levels=200, cmap='jet', norm=v_norm)
    ax2.set_title("Predict_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    img_name = os.path.join(filename, 'gif_make')
    os.makedirs(img_name, exist_ok=True)
    plt.savefig(img_name + '/--time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')


def val(filename_raw_data, model_path, L, U, xy_range):
    # 预测值
    data, total_data, total_data_no_state, bound_data, t_mean, t_std = load_data(filename_raw_data, L, U)

    x_mean = (xy_range[0] / L + xy_range[1] / L) / 2
    y_mean = (xy_range[2] / L + xy_range[3] / L) / 2
    x_std = ((xy_range[1] / L - xy_range[0] / L) ** 2 / 12) ** (1 / 2)
    y_std = ((xy_range[3] / L - xy_range[2] / L) ** 2 / 12) ** (1 / 2)

    data_mean = np.array([x_mean, y_mean, t_mean])
    data_std = np.array([x_std, y_std, t_std])

    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(model_path).items()})

    if isinstance(pinn_net, torch.nn.DataParallel):
        pinn_net = pinn_net.module

    pinn_net = pinn_net.to(device)

    raw_val = torch.tensor(total_data, dtype=torch.float32).to(device)

    L2_u, L2_v, L2_p = validation_2d(raw_val, pinn_net, L, U, rou=1000)

    print(f"L2_u: {L2_u}")
    print(f"L2_v: {L2_v}")
    print(f"L2_p: {L2_p}")


file_name = os.path.join('log', args.name)
train_cfg = jy_deal(os.path.join(file_name, 'train_cfg.yaml'), 'r')
filename_raw_data = train_cfg['data_path']
layer_mat = train_cfg['layer_mat']
model_name = 'best_' + train_cfg['model_name'] + '.pth'
L = train_cfg['L']
U = train_cfg['U']
Re = train_cfg['Re']
xy_range = train_cfg['xy_range']
rou = 1000
model_path = os.path.join(file_name, model_name)
val(filename_raw_data, model_path, L, U, xy_range)
