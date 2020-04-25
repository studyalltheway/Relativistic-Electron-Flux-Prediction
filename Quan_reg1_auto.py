#  Quan_reg1.py
import numpy as np
from Quan_reg1 import main
from config import config
import time
T0 = time.time()

if __name__ == '__main__':
    config = config()
    for j in range(10):
        config.seed += 1
        tau_range = np.arange(0.05, 1, 0.05)
        for i in range(19):
            config.run_num += 1
            config.tau = tau_range[i]

            main(config)
            print('>>>>>>>\n>>>>>>>>>>>', i)
    TT = time.time()
    print('final time>>>>>>>>>>>>>>', TT-T0)


