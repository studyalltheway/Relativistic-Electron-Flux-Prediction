#  save the prediction results of the best model into files
#  FFN_model_save.py
from FFN_model_subset import main
from config_subset import config
import time
T0 = time.time()

if __name__ == '__main__':
    config = config()
    for i in range(5):
        config.run_num += 1
        config.seed += 1
        #config.epochs += 200
        #config.node1 += 5
        #config.node2 += 5
        main(config)
        print('>>>>>>>\n>>>>>>>>>>>', i)
    TT = time.time()
    print('final time>>>>>>>>>>>>>>', TT-T0)


