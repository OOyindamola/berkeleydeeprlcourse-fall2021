python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 1000 \
--do_dagger --expert_data cs285/scripts/data/Ant-v2_data_250_rollouts.pkl \
--video_log_freq -1 --size 64
