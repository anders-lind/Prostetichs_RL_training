import torch as th
from rl_train.train.policies.network_index_handler import NetworkIndexHandler
import json

cfg_path = "rl_train/train/train_configs/my_imitation_OSL_angle_22_separated_net_partial_obs.json"
with open(cfg_path, "r") as f:
    cfg = json.load(f)
net_indexing_info = cfg["policy_params"]["custom_policy_params"]["net_indexing_info"]

obs = th.arange(40.).reshape(1,40)
handler = NetworkIndexHandler(net_indexing_info, None, None)
print("human obs num:", handler.get_observation_num("human_actor"))
print("exo obs num:", handler.get_observation_num("exo_actor"))
print("common critic obs num:", handler.get_observation_num("common_critic"))
print("map shapes:", handler.map_observation_to_network(obs, "human_actor").shape,
                   handler.map_observation_to_network(obs, "exo_actor").shape,
                   handler.map_observation_to_network(obs, "common_critic").shape)