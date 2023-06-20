import torch

import utils.pytorch_util as ptu
from torch.distributions import Uniform


def get_greedy_exploration_action(ob_np, policy=None, qfs=None, sample_size=32, sample_range=1, beta=None):

    assert ob_np.ndim == 1

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    _, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    begin = pre_tanh_mu_T-sample_range*std
    end = pre_tanh_mu_T+sample_range*std
    actions = torch.tanh(Uniform(begin, end).sample([sample_size]))
    # if batch==0 or batch==700 or batch==1400:
    #     x = torch.arange(-1,1,.005).to(ptu.device)
    #     gx, gy= torch.meshgrid(x,x)
    #     test_action = torch.cat((gx.reshape(-1,1), gy.reshape(-1,1)),1)
    #     test_data = [ob.reshape(1, -1).expand(test_action.size(0), len(ob)), test_action]
    #     Q1 = qfs[0](*test_data)
    #     Q2 = qfs[1](*test_data)
    #     torch.save(Q1, '/home/chenxing/tmp/inputs_Q1_'+str(batch)+'.t')
    #     torch.save(Q2, '/home/chenxing/tmp/inputs_Q2_'+str(batch)+'.t')
    #     # test
    #     ab = torch.tensor((-1.0,-1.0)).reshape(1,-1).cuda()
    #     test_data = [ob.reshape(1, -1).expand(ab.size(0), len(ob)), ab]
    #     qfs[0](*test_data)

    # if batch == 1400:
    #     exit()
    args =[ob.reshape(1,-1).expand(sample_size,len(ob)), actions]
    # args = list(torch.unsqueeze(i, dim=0) for i in (ob, actions[0]))
    Q1 = qfs[0](*args)
    Q2 = qfs[1](*args)

    Greedy_Q = torch.max(Q1,Q2).squeeze()
    Greedy_Q = Greedy_Q*beta
    wise_minus = Greedy_Q-Greedy_Q.max()
    log_sum = wise_minus.exp().sum().log()
    # input_tensor = Greedy_Q.exp() + 1e-8
    prob = (wise_minus - log_sum).exp()

    index_ac = prob.multinomial(1).item()

    max_q_ac = actions[index_ac]

    # mean_ac = torch.tanh(pre_tanh_mu_T)

    ac_np = ptu.get_numpy(max_q_ac)
    # print(max(max_q_ac), min(max_q_ac), max(mean_ac), min(mean_ac) )
    # print(max_q_ac- mean_ac)

    return ac_np, {}

