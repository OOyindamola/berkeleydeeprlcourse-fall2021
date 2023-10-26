# GCLAgent

python cs285/scripts/run_hw2_airl.py --env_name HalfCheetah-v2  -n 400 \
--train_batch_size 10000 --exp_name ppo --ep_len 150 \
--num_agent_train_steps_per_iter 80 --num_disc_train_steps_per_iter 40 \
--expert_policy 'expert_data_HalfCheetah-v2_23-09-2020_13-58-34_r-10.0.npz'
