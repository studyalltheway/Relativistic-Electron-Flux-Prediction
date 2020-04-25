# superprocess in order to find the best parameters
#  gridsearch method
#  MLP_model.py
from MLP_model import main
from config import config
import time
T0 = time.time()

if __name__ == '__main__':
    config = config()
    #reg = [0.00001, 0.0001, 0.001, 0.01]
    for i in range(5):
        config.run_num += 1
        #config.learnrate += 0.002
        config.seed += 1
        #config.batch_size += 50
        #config.node1 -= 10
        #config.node2 += 5
        #config.reg2 = reg[i]
        main(config)
        print('>>>>>>>\n>>>>>>>>>>>', i)
    TT = time.time()
    print('final time>>>>>>>>>>>>>>', TT-T0)


