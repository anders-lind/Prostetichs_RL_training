This project aims to use MyoAssist to train and compare different reinforcement learning algorithms for use in prosthetics.

Changes have been made to the original repository to use different reinforcement learning models and prosthetic models.

Read README_MyoAssist.md for the installation guide for MyoAssist and a overview of the MyoAssist project


- For the results of the project see the results folder
- The main branch contains the default PPO implementation
- The A2Cv2 branch contains the A2C implementation
- The SACv2 branch contains the SAC implementation


To train this model use:

python rl_train/run_train.py --config_file_path rl_train/train/train_configs/my_imitation_OSL_ankle_22_separated_net_partial_obs.json
