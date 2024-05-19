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
from utils import jy_deal, no_state, load_data

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Re3900_2024-05-18 11h 39m 57s')
parser.add_argument('--fps', type=int, default=100)
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


def compare_unsteady(file_name, filename_raw_data, model_path, L, U, xy_range):
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
    pinn_net.eval()

    net_inp = torch.tensor(total_data_no_state[:, 0:3], dtype=torch.float32).to(device)

    pre = pinn_net(net_inp)

    u_pre = pre[:, 0].reshape(-1, 1)
    v_pre = pre[:, 1].reshape(-1, 1)
    p_pre = pre[:, 2].reshape(-1, 1)

    # 处理数据
    x_raw_mat = total_data[:, 0].reshape(-1, 1)
    y_raw_mat = total_data[:, 1].reshape(-1, 1)
    t_raw_mat = total_data[:, 2].reshape(-1, 1)
    u_raw_mat = total_data[:, 3].reshape(-1, 1)
    v_raw_mat = total_data[:, 4].reshape(-1, 1)
    p_raw_mat = total_data[:, 5].reshape(-1, 1)

    u_pre_mat = u_pre.cpu().detach().numpy()
    v_pre_mat = v_pre.cpu().detach().numpy()
    p_pre_mat = p_pre.cpu().detach().numpy()

    u_pre_mat, v_pre_mat, p_pre_mat = fan_wulianggang(u_pre_mat, v_pre_mat, p_pre_mat, U, rou)

    temp = np.concatenate((x_raw_mat, y_raw_mat, t_raw_mat, total_data[:, 3:]), 1)
    min_data = np.min(temp, axis=0).reshape(1, -1)
    max_data = np.max(temp, axis=0).reshape(1, -1)
    x_unique_total = np.unique(x_raw_mat).reshape(-1, 1)
    y_unique_total = np.unique(y_raw_mat).reshape(-1, 1)
    t_unique_total = np.unique(t_raw_mat).reshape(-1, 1)
    time_series = t_unique_total[:, 0].reshape(-1, 1)

    mesh_x, mesh_y = np.meshgrid(x_unique_total, y_unique_total)
    x_selected_total = x_raw_mat[t_raw_mat == time_series[0].item()]
    y_selected_total = y_raw_mat[t_raw_mat == time_series[0].item()]
    ind_total = [[], [], []]
    for i in range(len(x_selected_total)):
        x_ind = np.where(x_unique_total == x_selected_total[i])[0][0]
        y_ind = np.where(y_unique_total == y_selected_total[i])[0][0]
        ind_total[1].append(x_ind)
        ind_total[0].append(y_ind)
        ind_total[2].append(i)

    v_norm_u = matplotlib.colors.Normalize(vmin=min_data[0, 3], vmax=max_data[0, 3])
    v_norm_v = matplotlib.colors.Normalize(vmin=min_data[0, 4], vmax=max_data[0, 4])
    v_norm_p = matplotlib.colors.Normalize(vmin=min_data[0, 5], vmax=max_data[0, 5])

    for select_time in tqdm(time_series):
        time = select_time.item()
        u_selected_ = u_raw_mat[t_raw_mat == select_time]
        u_selected = np.zeros_like(mesh_x)
        u_selected[ind_total[0], ind_total[1]] = u_selected_[ind_total[2]]

        v_selected_ = v_raw_mat[t_raw_mat == select_time]
        v_selected = np.zeros_like(mesh_x)
        v_selected[ind_total[0], ind_total[1]] = v_selected_[ind_total[2]]

        p_selected_ = p_raw_mat[t_raw_mat == select_time]
        p_selected = np.zeros_like(mesh_x)
        p_selected[ind_total[0], ind_total[1]] = p_selected_[ind_total[2]]

        u_predicted_ = u_pre_mat[t_raw_mat == select_time]
        u_predicted = np.zeros_like(mesh_x)
        u_predicted[ind_total[0], ind_total[1]] = u_predicted_[ind_total[2]]

        v_predicted_ = v_pre_mat[t_raw_mat == select_time]
        v_predicted = np.zeros_like(mesh_x)
        v_predicted[ind_total[0], ind_total[1]] = v_predicted_[ind_total[2]]

        p_predicted_ = p_pre_mat[t_raw_mat == select_time]
        p_predicted = np.zeros_like(mesh_x)
        p_predicted[ind_total[0], ind_total[1]] = p_predicted_[ind_total[2]]

        plot_compare_time_series(file_name, mesh_x, mesh_y, u_selected, u_predicted, time, v_norm_u, name='u')
        plot_compare_time_series(file_name, mesh_x, mesh_y, v_selected, v_predicted, time, v_norm_v, name='v')
        plot_compare_time_series(file_name, mesh_x, mesh_y, p_selected, p_predicted, time, v_norm_p, name='p')

    return [total_data_no_state[:, 0:3], u_pre_mat, v_pre_mat, p_pre_mat], time_series


def make_plot(train_cfg):
    model_name = train_cfg['model_name']
    write_path = train_cfg['write_path']
    save_interval = train_cfg['save_interval']
    loss_path = os.path.join(write_path, 'loss.csv')
    evaluate_path = os.path.join(write_path, 'evaluate.csv')
    loss_data = np.loadtxt(f"{loss_path}", delimiter=',')
    evaluate_data = np.loadtxt(f"{evaluate_path}", delimiter=',')
    loss_data = loss_data[:, 1:5]
    x = [i + 1 for i in range(0, save_interval * loss_data.shape[0], save_interval)]
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12.8, 9.6))
    plt.plot(x, loss_data)
    plt.legend(['total_loss', 'data_loss', 'eqa_loss', 'bound_loss'], fontsize=20)
    plt.title(model_name, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{write_path}/loss_process.png')

    plt.figure(figsize=(12.8, 9.6))
    plt.plot(x, evaluate_data)
    plt.legend(['L2_u', 'L2_v', 'L2_p'], fontsize=20)
    plt.title(model_name, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{write_path}/evaluate_process.png')


file_name = os.path.join('log', args.name)
train_cfg = jy_deal(os.path.join(file_name, 'train_cfg.yaml'), 'r')
make_plot(train_cfg)
filename_raw_data = train_cfg['data_path']
layer_mat = train_cfg['layer_mat']
model_name = 'best_' + train_cfg['model_name'] + '.pth'
L = train_cfg['L']
U = train_cfg['U']
Re = train_cfg['Re']
xy_range = train_cfg['xy_range']

output_file = f'pre_data_{Re}'
rou = 1000
model_path = os.path.join(file_name, model_name)
res, t = compare_unsteady(file_name, filename_raw_data, model_path, L, U, xy_range)

data = np.hstack(res)

scipy.io.savemat(f'{file_name}/{output_file}.mat', mdict={'stack': data})
df = pd.DataFrame(data)
df.to_csv(f"{file_name}/{output_file}.csv", index=False, header=False)
make_flow_gif(file_name, t, name='u', fps_num=args.fps)
make_flow_gif(file_name, t, name='v', fps_num=args.fps)
make_flow_gif(file_name, t, name='p', fps_num=args.fps)
