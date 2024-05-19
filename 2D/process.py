import multiprocessing
import time
from train2d import train
import torch

torch.multiprocessing.set_start_method('spawn', force=True)

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=train,
                                 args=('data/data_Re3900.mat', 3900, 0.1, 1))
    p2 = multiprocessing.Process(target=train,
                                 args=('data/data_Re3900_2.mat', 3900, 0.1, 1))
    p1.start()
    p2.start()
    time.sleep(10)
    p1.join()
    p2.join()
