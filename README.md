# NS_PINN

基于物理信息神经网络PINN，对NS方程进行反演

对数据监督损失、NS方程损失、边界损失采用等权值损失计算

目前在二维圆柱绕流场景下，对于Re=3900的数据进行训练，得到效果图如下：





# 一、环境配置

首先需安装 python>=3.10.2，然后将项目移至全英文路径下,安装以下依赖

## pytorch

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```

## 其他依赖

```bash
pip install -r requirements.txt
```



# 二、案例复现

### 一、数据获取

在以下链接中获取样本数据：

将获取的mat文件放入2D/data路径下

### 二、模型训练

进入2D文件夹路径：

```bash
cd 2D
```

在命令行输入以下命令开始训练

```bash
python train2d.py --data_path 'data/data_Re3900_2.mat'
```

训练好的模型和过程记录保存在log文件夹下,按训练开始时间命名文件夹

如果需要断续续训，则使用以下命令：

```bash
python train2d.py --name "2024-04-30 13h 19m 50s"
```

传入需要续训的文件名即可，但注意会使用之前设置的训练参数，如需要修改训练参数则在续训之前到相应文件夹下对train_cfg.yaml进行修改

### 三、模型验证

可以对训练过程中保存的损失最小的模型进行验证

```bash
python val.py --name "2024-04-30 13h 19m 50s"
```

### 五、流场预测

```bash
python export_plot.py --name "2024-04-30 13h 19m 50s"
```

预测结果保存在log文件夹下对应训练文件夹中


