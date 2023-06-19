# DATA
dataset = 'CULane'
data_root = 'dataset/CULane'

# TRAIN
epoch = 100  # 50
batch_size = 32
optimizer = 'Adam'  # ['SGD','Adam']
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'  # ['multi', 'cos']
steps = [25, 38]  # 仅multi用到
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = False
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0  
shp_loss_w = 0.0

# EXP
note = ''

log_path = 'logs/CULane'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = 'tmp'

num_lanes = 4




