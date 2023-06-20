from collections import OrderedDict
import ray
ray.init(local_mode=True)
# import torch
# import copy
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(3, 1),
# )
# # tt =KK.remote()
# # futures = [tt.mkk.remote(OrderedDict()) for i in range(4)]
# # print(ray.get(futures)) # [0, 1, 4, 9]
# a = model.state_dict()
# ray.put(a)