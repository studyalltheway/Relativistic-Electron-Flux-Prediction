
class config():
    def __init__(self):
        self.path0 = './result/'
        self.datapath = './predata/'
        self.file0 = ''
        self.file1 = ''
        self.file2 = ''
        self.set1 = 'Set1.csv'
        self.set2 = 'Set2.csv'
        self.set3 = 'Set3.csv'
        self.set4 = 'Set4.csv'
        self.set5 = 'Set5.csv'
        self.label1 = 'Label1.csv'
        ###------------------------------------
        # log information, running ID
        # FFN: run_num: 1-1000
        # MLP, run_num: 1000-2000
        # quantile regression: 2000-3000

        # running ID
        self.run_num = 1
        # random seed (very important!, every experiment need running at least 5 times with different random seeds!)
        self.seed = 1
        # log information
        self.logfile = 'logfile.csv'
        self.logfile2 = 'logfile2.csv'
        self.subsetlog = 'subsetlog.csv'

        ########parameters of the neural networks
        # input number
        self.num_in = 85
        # the number of nodes in hidden layer1
        self.node1 = 25
        # the number of nodes in hidden layer2
        self.node2 = 25
        # the initial weights of the scaling transformation layer
        self.weight_init = 1
        ## activation function
        # the activation function of hidden layer 1
        self.fun1 = 'selu'  # tanh, sigmoid, relu, leaky_relu
        # the activation function of hidden layer 2
        self.fun2 = 'sigmoid'  # SELU, LEAKY_RELU(le_0.1), tanh, relu
        # Loss function
        self.Loss = 'L2'  # L1, merge(m4.5), tau(t0.5)

        ###### parameters of the training
        # number of training epochs
        self.epochs = 1600
        # training batch_size
        self.batch_size = 300
        # learning rate of Adam Optimizer
        self.learnrate = 0.012
        # L1 regularization
        self.reg1 = 0.001
        # L2 regularization
        self.reg2 = 0.001


        ##### parameters of quantile regression
        self.tau = 0.5
        #  three quantiles of multiple outputs of the quantile neural network
        self.taurange = [0.2, 0.5, 0.8]
        #  the weights set
        self.pterm = 0.1



        ### experiment 1 series：MLP experimet
        #  one hidden layer parameter
        self.F1_node = 30
        #  three hidden layers parameter
        self.F3_node1 = 50
        self.F3_node2 = 50
        self.F3_node3 = 20
        #  four hidden layers parameter
        self.F4_node1 = 100
        self.F4_node2 = 50
        self.F4_node3 = 50
        self.F4_node4 = 20


        return
