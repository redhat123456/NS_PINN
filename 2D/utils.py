import shutil
import sys

import numpy as np
import pandas as pd
import pynvml
import os
import matplotlib.pyplot as plt
import time
from pyDOE import lhs
import scipy
import torch
import yaml
from torch.autograd import grad





def gradient(y, x):
    return grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]


# grad_outputs=torch.ones_like(y)


def select_point(data, n, center_point=(0, 0), is_dis=True):
    x_filtered = data[:, 0]
    y_filtered = data[:, 1]
    if is_dis:
        dis_filtered = 1 / ((x_filtered - center_point[0]) ** 2 + (y_filtered - center_point[1]) ** 2)
        selected_indices = np.random.choice(len(data), size=n, p=dis_filtered / np.sum(dis_filtered), replace=False)
    else:
        selected_indices = np.random.choice(len(data), size=n, replace=False)
    # plt.scatter(data[selected_indices, 0], data[selected_indices, 1])
    # plt.show()
    selected_tensor = torch.tensor(data[selected_indices], dtype=torch.float32)
    return selected_tensor


def state(data, L, U, rou):
    T = L / U
    P = rou * U ** 2
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    t = data[:, 2].reshape(-1, 1)
    u = data[:, 3].reshape(-1, 1)
    v = data[:, 4].reshape(-1, 1)
    p = data[:, 5].reshape(-1, 1)
    x_ = x * L
    y_ = y * L
    t_ = t * T
    u_ = u * U
    v_ = v * U
    p_ = p * P
    r_data = np.hstack((x_, y_, t_, u_, v_, p_))
    return r_data


def no_state(data, L, U, rou):
    T = L / U
    P = rou * U ** 2
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    t = data[:, 2].reshape(-1, 1)
    u = data[:, 3].reshape(-1, 1)
    v = data[:, 4].reshape(-1, 1)
    p = data[:, 5].reshape(-1, 1)
    x_ = x / L
    y_ = y / L
    t_ = t / T
    u_ = u / U
    v_ = v / U
    p_ = p / P
    r_data = np.hstack((x_, y_, t_, u_, v_, p_))
    return r_data


def validation_2d(data, out_mean, out_std, model, L, U, device, rou=1000):
    data = torch.tensor(data, dtype=torch.float32).to(device)
    out_mean = torch.tensor(out_mean, dtype=torch.float32).to(device)
    out_std = torch.tensor(out_std, dtype=torch.float32).to(device)
    # 预测值
    inp_data = data[:, :3]
    inp_data[:, 0] = inp_data[:, 0] / L
    inp_data[:, 1] = inp_data[:, 1] / L
    inp_data[:, 2] = inp_data[:, 2] / (L / U)
    model.eval()
    with torch.no_grad():
        pre = model(inp_data)
        pre = pre * out_std + out_mean

    u_pre = pre[:, 0]
    v_pre = pre[:, 1]
    p_pre = pre[:, 2]

    u_raw_mat = data[:, 3].cpu().numpy()
    v_raw_mat = data[:, 4].cpu().numpy()
    p_raw_mat = data[:, 5].cpu().numpy()

    u_pre_mat = u_pre.cpu().detach().numpy()
    v_pre_mat = v_pre.cpu().detach().numpy()
    p_pre_mat = p_pre.cpu().detach().numpy()

    P = rou * U ** 2
    u_pre_mat = u_pre_mat * U
    v_pre_mat = v_pre_mat * U
    p_pre_mat = p_pre_mat * P

    # 处理数据
    L2_u = (np.linalg.norm(u_pre_mat - u_raw_mat) / np.linalg.norm(u_raw_mat))
    L2_v = (np.linalg.norm(v_pre_mat - v_raw_mat) / np.linalg.norm(v_raw_mat))
    L2_p = (np.linalg.norm(p_pre_mat - p_raw_mat) / np.linalg.norm(p_raw_mat))

    return L2_u, L2_v, L2_p


def build_optimizer(network, learning_rate):
    # default 默认优化器
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # 超参数搜索优化器

    return optimizer, scheduler


def pre_train_loading_2d(filename_data):
    # load data points(only once)
    # 加载真实数据点(仅一次)
    x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts, data_mean, data_std = load_data_points_2d(filename_data)
    if x_sub_ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts], 1)
        true_dataset = data_sub[torch.randperm(data_sub.size(0))]  # 乱序
    else:
        true_dataset = None
    # load collocation points(only once)
    # 加载方程点(仅一次)
    return true_dataset, data_mean, data_std


def load_data(filename, L, U):
    data_mat = scipy.io.loadmat(filename)
    data = data_mat['data']
    total_data = data_mat['total']
    bound_data = data_mat['bound']
    data = no_state(data, L, U, 1000)
    total_data_no_state = no_state(total_data, L, U, 1000)
    bound_data = no_state(bound_data, L, U, 1000)
    data_mean = np.mean(total_data_no_state, axis=0)
    data_std = np.std(total_data_no_state, axis=0)
    return data, total_data, total_data_no_state, bound_data, data_mean, data_std


def load_data_points_2d(filename, L, U, state=False):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    if not state:
        stack = no_state(stack, L, U, 1000)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    low_bound = np.array([np.min(x), np.min(y), np.min(t)]).reshape(1, -1)
    up_bound = np.array([np.max(x), np.max(y), np.max(t)]).reshape(1, -1)
    temp = np.concatenate((x, y, t), 1)
    data_mean = np.mean(temp, axis=0).reshape(1, -1)
    data_std = np.std(temp, axis=0).reshape(1, -1)
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.min(temp, 0)
    feature_mat[1, :] = np.max(temp, 0)
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound, data_mean, data_std


def load_equation_points_lhs(low_bound, up_bound, dimension, points, radius, center_point=(0, 0)):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    Eqa_points = torch.from_numpy(eqa_xyzt).float()
    Eqa_points = Eqa_points[torch.randperm(Eqa_points.size(0))]
    xy_cord = Eqa_points[:, 0:2]
    true_ind = torch.where(
        (xy_cord[:, 0] - center_point[0]) ** 2 + (xy_cord[:, 1] - center_point[1]) ** 2 > radius ** 2)
    eqa_dataset = Eqa_points[true_ind]
    return eqa_dataset


def boundary_points_lhs(low_bound, up_bound, points, r):
    theta_points = 2 * np.pi * lhs(1, points)
    t_points = (up_bound[:, 2] - low_bound[:, 2]) * lhs(1, points) + low_bound[:, 2]
    x_points = r * np.cos(theta_points)
    y_points = r * np.sin(theta_points)
    boundary_points = np.hstack((x_points, y_points, t_points))
    boundary_points = torch.from_numpy(boundary_points).float()
    boundary_points = boundary_points[torch.randperm(boundary_points.size(0))]
    return boundary_points


def low_points_lhs(low_bound, up_bound, points):
    t_points = (up_bound[:, 2] - low_bound[:, 2]) * lhs(1, points) + low_bound[:, 2]
    x_points = np.ones((points, 1)) * low_bound[:, 0]
    y_points = (up_bound[:, 1] - low_bound[:, 1]) * lhs(1, points) + low_bound[:, 1]
    low_points = np.hstack((x_points, y_points, t_points))
    low_points = torch.from_numpy(low_points).float()
    low_points = low_points[torch.randperm(low_points.size(0))]
    return low_points


def up_points_lhs(low_bound, up_bound, points):
    t_points = (up_bound[:, 2] - low_bound[:, 2]) * lhs(1, points) + low_bound[:, 2]
    x_points = np.ones((points, 1)) * up_bound[:, 0]
    y_points = (up_bound[:, 1] - low_bound[:, 1]) * lhs(1, points) + low_bound[:, 1]
    up_points = np.hstack((x_points, y_points, t_points))
    up_points = torch.from_numpy(up_points).float()
    up_points = up_points[torch.randperm(up_points.size(0))]
    return up_points


def pre_train_loading_2d_normal(filename_data, L, U):
    x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts, low_bound, up_bound, data_mean, data_std = load_data_points_2d(
        filename_data, L, U)

    if x_sub_ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts], 1)
        true_dataset = data_sub[torch.randperm(data_sub.size(0))]  # 乱序
    else:
        true_dataset = None

    return true_dataset, low_bound, up_bound, data_mean, data_std


def remove_file(old_path, new_path):
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)


def get_label_list(imgpath):
    file_path = f'./{imgpath}'

    path_list = []

    for i in os.walk(file_path):
        path_list.append(i)

    label_dict = dict()
    label_name_list = []
    label_list = []

    for i in range(len(path_list[0][1])):
        label = path_list[0][1][i]
        label_dict[label] = path_list[i + 1][2]

    for i in label_dict.keys():
        label_list.append(i)
        for j in label_dict[i]:
            label_name_list.append([i, j])

    return label_name_list, label_dict, label_list


def add_log(txt, txt_list, is_print=True):
    if is_print:
        print(txt)
    txt_list.append(txt + '\r\n')


def write_log(in_path, filename, txt_list):
    os.makedirs(in_path, exist_ok=True)
    path = os.path.join(in_path, filename + '.txt')
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


def get_gpu_usage(gpu_id):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_utilization = util.gpu
        memory_use = '%.2f' % (mem_info.used / (1024 ** 3))
        memory_total = '%.2f' % (mem_info.total / (1024 ** 3))
        return gpu_utilization, memory_use, memory_total
    except pynvml.NVMLError as error:
        print(f"Failed to get GPU usage: {error}")
        return None, None, None


def train_bar(args):
    pynvml.nvmlInit()
    t_epoch, t_inner, start, epoch, inner, train_loss, start_epoch = args
    # 获取GPU句柄，这里假设你只有一个GPU
    l = 30
    f_p = epoch / t_epoch
    f_n = int(f_p * l)
    finsh = "▓" * f_n
    need_do = "-" * (l - f_n)
    e_process = finsh + need_do + '|'

    l2 = 15
    f_p_b = inner / t_inner
    f_n = int(f_p_b * l2)
    finsh_b = "▓" * f_n
    need_b = "-" * (l2 - f_n)
    batch_process = finsh_b + need_b + '|'

    dur = time.perf_counter() - start
    finish_epoch = epoch - 1 + f_p_b
    res = (t_epoch - start_epoch - finish_epoch) / ((finish_epoch - start_epoch) / dur)

    # 获取GPU利用率和内存占用率
    gpu_util, gpu_mem_use, gpu_mem_total = get_gpu_usage(0)
    gpu_mem_util = str(gpu_mem_use) + '/' + str(gpu_mem_total) + 'GB'
    gpu_util = str(gpu_util) + '%'

    epochs = str(epoch) + '/' + str(t_epoch)
    proc = "\r{:10s} {:14s} {:10s} {:.06e} {:16s} {:31s} {:.2f}s/{:.2f}s".format(
        epochs, gpu_mem_util, gpu_util, train_loss, batch_process, e_process, dur, res)
    # print(proc, end='')
    sys.stdout.write(proc)
    sys.stdout.flush()
    # time.sleep(0.01)
    if epoch == t_epoch and inner == t_inner:
        print()


def bar(i, total, start, des):
    l = 30
    f_p = i / total
    n_p = (total - i) / total
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    res = (total - i) / (i / dur)
    proc = "\r{}({}/{}):{:^3.2f}%[{}->{}] time:{:.2f}s/{:.2f}s".format(
        des, i, total, progress, finsh, need_do, dur, res)
    sys.stdout.write(proc)
    sys.stdout.flush()
    if i == total:
        print()


def ini_env():
    import torch
    import platform
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        device_name = []
        for i in range(gpu_num):
            device_name.append(str(i) + '  ' + torch.cuda.get_device_name(i))
        device = torch.device('cuda')
        gpus = [i for i in range(gpu_num)]
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        device_name = ['cpu']
        device = torch.device('cpu')
        gpus = []
    os_name = str(platform.system())
    return device, gpus, device_name


def make_dir(path):
    import os
    base_path = os.path.dirname(path)
    if base_path:
        os.makedirs(base_path, exist_ok=True)


def jy_deal(path, mode, dic=None):
    make_dir(path)
    jy = path.split('.')[-1]
    if jy == 'json':
        import json
        if mode == 'w':
            with open(path, mode, encoding="utf-8") as f:
                json.dump(dic, f)
        elif mode == 'r':
            with open(path, mode, encoding='UTF-8') as f:
                res = json.load(f)
            return res
        else:
            raise ValueError(f'mode {mode} not supported')
    elif jy == 'yaml':

        if mode == 'w':
            with open(path, mode, encoding="utf-8") as f:
                yaml.dump(dic, f)
        elif mode == 'r':
            with open(path, mode, encoding='utf-8') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
            return res
        else:
            raise ValueError(f'mode {mode} not supported')
    else:
        raise ValueError('invalid file')


def save_csv(path, *args):
    args_list = []
    for i in args:
        args_list.append(np.array(i).reshape(1, 1))
    t_data = np.hstack(args_list).reshape(1, -1)
    loss_save = pd.DataFrame(t_data)
    loss_save.to_csv(path, index=False, header=False, mode='a')
