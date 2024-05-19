import argparse
import os
import shutil

import imageio
import matplotlib
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/data_Re3900_2.mat')
parser.add_argument('--FPS', type=int, default=100)
args = parser.parse_args()


def plot_compare_time_series(filename, data_total, bound_total, select_time, v_norm_total, v_norm_bound, name='q'):
    x_total, y_total, q_total = data_total
    x_bound, y_bound, q_bound = bound_total
    plt.cla()
    # v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = ax1.contourf(x_total, y_total, q_total, levels=200, cmap='jet', norm=v_norm_total)
    fig.colorbar(mappable, cax=cbar_ax)
    ax1.set_title("Total_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    ax2.contourf(x_bound, y_bound, q_bound, levels=200, cmap='jet', norm=v_norm_bound)
    ax2.set_title("Bound_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    os.makedirs(filename, exist_ok=True)
    plt.savefig(filename + '/--time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')


def plot_load_data(filename):
    data_mat = scipy.io.loadmat(filename)
    data = data_mat['data']
    total_data = data_mat['total']
    bound_data = data_mat['bound']
    return data, total_data, bound_data


def compare_unsteady(filename):
    # 预测值
    _, total_data, bound_data = plot_load_data(filename)

    x_total = total_data[:, 0].reshape(-1, 1)
    y_total = total_data[:, 1].reshape(-1, 1)
    t_total = total_data[:, 2].reshape(-1, 1)
    u_total = total_data[:, 3].reshape(-1, 1)
    v_total = total_data[:, 4].reshape(-1, 1)
    p_total = total_data[:, 5].reshape(-1, 1)

    x_bound = bound_data[:, 0].reshape(-1, 1)
    y_bound = bound_data[:, 1].reshape(-1, 1)
    t_bound = bound_data[:, 2].reshape(-1, 1)
    u_bound = bound_data[:, 3].reshape(-1, 1)
    v_bound = bound_data[:, 4].reshape(-1, 1)
    p_bound = bound_data[:, 5].reshape(-1, 1)

    # 处理数据
    t_unique_bound = np.unique(t_bound).reshape(-1, 1)
    x_unique_bound = np.unique(x_bound).reshape(-1, 1)
    y_unique_bound = np.unique(y_bound).reshape(-1, 1)
    time_series = t_unique_bound[:, 0].reshape(-1, 1)

    mesh_x_bound, mesh_y_bound = np.meshgrid(x_unique_bound, y_unique_bound)
    x_selected_bound = x_bound[t_bound == time_series[0].item()]
    y_selected_bound = y_bound[t_bound == time_series[0].item()]
    ind_bound = [[], [], []]
    for i in range(len(x_selected_bound)):
        x_ind = np.where(x_unique_bound == x_selected_bound[i])[0][0]
        y_ind = np.where(y_unique_bound == y_selected_bound[i])[0][0]
        ind_bound[1].append(x_ind)
        ind_bound[0].append(y_ind)
        ind_bound[2].append(i)

    x_unique_total = np.unique(x_total).reshape(-1, 1)
    y_unique_total = np.unique(y_total).reshape(-1, 1)
    t_unique_total = np.unique(t_total).reshape(-1, 1)
    time_series = t_unique_total[:, 0].reshape(-1, 1)

    shutil.rmtree('ini_plot')
    mesh_x_total, mesh_y_total = np.meshgrid(x_unique_total, y_unique_total)
    x_selected_total = x_total[t_total == time_series[0].item()]
    y_selected_total = y_total[t_total == time_series[0].item()]
    ind_total = [[], [], []]
    for i in range(len(x_selected_total)):
        x_ind = np.where(x_unique_total == x_selected_total[i])[0][0]
        y_ind = np.where(y_unique_total == y_selected_total[i])[0][0]
        ind_total[1].append(x_ind)
        ind_total[0].append(y_ind)
        ind_total[2].append(i)

    time_series = t_unique_bound[:, 0].reshape(-1, 1)

    v_norm_u_total = matplotlib.colors.Normalize(vmin=np.min(u_total), vmax=np.max(u_total))
    v_norm_v_total = matplotlib.colors.Normalize(vmin=np.min(v_total), vmax=np.max(v_total))
    v_norm_p_total = matplotlib.colors.Normalize(vmin=np.min(p_total), vmax=np.max(p_total))
    v_norm_u_bound = matplotlib.colors.Normalize(vmin=np.min(u_bound), vmax=np.max(u_total))
    v_norm_v_bound = matplotlib.colors.Normalize(vmin=np.min(v_bound), vmax=np.max(v_bound))
    v_norm_p_bound = matplotlib.colors.Normalize(vmin=np.min(p_bound), vmax=np.max(p_bound))

    for select_time in tqdm(time_series):
        time = select_time.item()
        u_selected_bound = u_bound[t_bound == select_time]
        u_bound_v = np.zeros_like(mesh_x_bound)
        u_bound_v[ind_bound[0], ind_bound[1]] = u_selected_bound[ind_bound[2]]

        v_selected_bound = v_bound[t_bound == select_time]
        v_bound_v = np.zeros_like(mesh_x_bound)
        v_bound_v[ind_bound[0], ind_bound[1]] = v_selected_bound[ind_bound[2]]

        p_selected_bound = p_bound[t_bound == select_time]
        p_bound_v = np.zeros_like(mesh_x_bound)
        p_bound_v[ind_bound[0], ind_bound[1]] = p_selected_bound[ind_bound[2]]

        u_selected_total = u_total[t_total == select_time]
        u_total_v = np.zeros_like(mesh_x_total)
        u_total_v[ind_total[0], ind_total[1]] = u_selected_total[ind_total[2]]

        v_selected_total = v_total[t_total == select_time]
        v_total_v = np.zeros_like(mesh_x_total)
        v_total_v[ind_total[0], ind_total[1]] = v_selected_total[ind_total[2]]

        p_selected_total = p_total[t_total == select_time]
        p_total_v = np.zeros_like(mesh_x_total)
        p_total_v[ind_total[0], ind_total[1]] = p_selected_total[ind_total[2]]

        plot_compare_time_series('ini_plot', [mesh_x_total, mesh_y_total, u_total_v],
                                 [mesh_x_bound, mesh_y_bound, u_bound_v], time, v_norm_u_total,
                                 v_norm_u_bound,
                                 name='u')
        plot_compare_time_series('ini_plot', [mesh_x_total, mesh_y_total, v_total_v],
                                 [mesh_x_bound, mesh_y_bound, v_bound_v], time, v_norm_v_total,
                                 v_norm_v_bound,
                                 name='v')
        plot_compare_time_series('ini_plot', [mesh_x_total, mesh_y_total, p_total_v],
                                 [mesh_x_bound, mesh_y_bound, p_bound_v], time, v_norm_p_total,
                                 v_norm_p_bound,
                                 name='p')

    return time_series


def make_flow_gif(in_path, out_path, time_series, name='q', fps_num=5):
    gif_images = []
    gif_name = os.path.join(out_path, name + '.gif')
    for select_time in time_series:
        time = select_time.item()
        gif_images.append(imageio.v2.imread(in_path + '/--time' + "{:.2f}".format(time) + name + '.png'))
    imageio.mimsave(gif_name, gif_images, fps=fps_num, duration=0.1, loop=0)


t = compare_unsteady(args.data_path)
make_flow_gif('ini_plot', 'ini_plot', t, name='u', fps_num=args.fps)
make_flow_gif('ini_plot', 'ini_plot', t, name='v', fps_num=args.fps)
make_flow_gif('ini_plot', 'ini_plot', t, name='p', fps_num=args.fps)
