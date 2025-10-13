from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
import logging

logging.basicConfig(level=logging.DEBUG)

# # Origional 2D model with 22 muscles
# env = MyoAssistLegBase(model_path="models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml",
#                        env_params=TrainSessionConfigBase.EnvParams())

# # myoLeg26_TUTORIAL - Works
# env = MyoAssistLegBase(model_path="models/26muscle_3D/myoLeg26_TUTORIAL.xml",
#                        env_params=TrainSessionConfigBase.EnvParams())

# # myoLeg22_2D_OSL_A - Now works!
env = MyoAssistLegBase(model_path="models/22muscle_2D/myoLeg22_2D_OSL_A.xml", 
                       env_params=TrainSessionConfigBase.EnvParams())

# myoLeg26_TUTORIAL - Now works!
# env = MyoAssistLegBase(model_path="models/26muscle_3D/myoLeg26_OSL_A.xml",
#                        env_params=TrainSessionConfigBase.EnvParams())

env.mujoco_render_frames = True

obs, info = env.reset(seed=1)
for timestep in range(150):
    # random_action = env.action_space.sample() # Random action
    random_action = env.action_space.sample() * 0.0  # Zero action
    obs, reward, done, truncated, info = env.step(random_action)

