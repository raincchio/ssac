import torch

from utils.env_utils import env_producer
from component.path_collector import MdpPathCollector
import utils.eval_util as eval_util
from model.agent import get_policy_producer,get_q_producer
from utils.default import variant, get_cmd_args
from collections import OrderedDict
from utils.eval_util import create_stats_ordered_dict
from utils.pytorch_util import set_gpu_mode

args = get_cmd_args()
domain = 'swimmer'
seed = 1

set_gpu_mode(args.use_gpu and torch.cuda.is_available(), seed)


expl_env = env_producer(domain, seed)
obs_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

path_collector = MdpPathCollector(expl_env,)

M = variant['layer_size']
q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M, M])
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M, M])

policy = policy_producer()
qf1 = q_producer()
qf2 = q_producer()
model = 'epoch_1999'
path = '/home/chenxing/experiments/model/'+model+'.ml'
model_state_dict = torch.load(path)
policy.load_state_dict(model_state_dict['policy_state_dict'])
qf1.load_state_dict(model_state_dict['qf1_state_dict'])
qf2.load_state_dict(model_state_dict['qf2_state_dict'])

expl_paths = path_collector.collect_new_paths(
    policy,max_path_length=1000,num_steps=5000,discard_incomplete_paths=False
)

path_lens = [len(path['actions']) for path in expl_paths]
stats = OrderedDict([
    ('num steps total', sum(path_lens)),
    ('num paths total', len(path_lens)),
])
stats.update(create_stats_ordered_dict(
    "path length",
    path_lens,
    always_show_all_stats=True,
))
print(stats)

info = eval_util.get_generic_path_information(expl_paths)
for k,v in info.items():
    print(k,v)
