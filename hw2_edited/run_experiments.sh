# GCLAgent

python cs285/scripts/run_hw2_gcl.py --env_name CartPole-v0 -acs 1 -n 400 -b 100 \
--train_batch_size 100 -rtg -dsa --exp_name lb_rtg_dsa  --save_params \
--num_agent_train_steps_per_iter 10 --num_reward_train_steps_per_iter 10 \
--discount 1.0 --reward_learning_rate 0.001 \
--expert_policy '/home/oyindamola/Research/homework_fall2021/hw3/data/q4_100_1_CartPole-v0_22-02-2022_11-45-22/1/agent_itr_80.pt' \

#
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 100 \
# -rtg -dsa --exp_name q1_sb_no_rtg_dsa --num_agent_train_steps_per_iter 10 \
#   --save_params --train_batch_size 100 --discount 1.0 #&&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -rtg -dsa --exp_name q1_sb_rtg_dsa  --num_agent_train_steps_per_iter 100 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -rtg --exp_name q1_sb_rtg_na  --num_agent_train_steps_per_iter 100 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -dsa --exp_name q1_lb_no_rtg_dsa  --num_agent_train_steps_per_iter 100 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -rtg -dsa --exp_name q1_lb_rtg_dsa  --num_agent_train_steps_per_iter 100 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -rtg --exp_name q1_lb_rtg_na  --num_agent_train_steps_per_iter 100

#inverted pendulum

# echo "==================== InvertedPendulum =======================" ####
# for b in 5000 10000 20000
# do
#     for l in 0.005 0.01 0.02
#     do
#           echo -n "Batch - $b, Learning rate - $l "
#           python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
#           --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $l -rtg \
#           --exp_name q2_b${b}_lr${l} --seed 10 20 30
#     done
#
#   echo "" #### print the new line ###
# done
#
# ## inverted nn_baseline
# python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 10000 -lr 0.01 --reward_to_go --nn_baseline --exp_name q2_baseline_b10000_r0.005 --seed 10 20 30
#
#
#
# python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.01 -rtg \
# --exp_name q2_b10000_lr0.01_final --seed 10 20 30
# # LunarLander
# &&
# echo "==================== LunarLander =======================" ####
# python cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \ --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005 --seed 1 5
#
# &&
#
# # HalfCheetah
#
# echo "==================== HalfCheetah =======================" ####
# for b in 10000 30000 50000
# do
#     for l in 0.005 0.01 0.02
#     do
#           echo -n "Batch - $b, Learning rate - $l "
#           python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
#           --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $l -rtg --nn_baseline \
#           --exp_name q4_search_b${b}_lr${l}_rtg_nnbaseline --seed 10 20 30
#     done
#
#   echo "" #### print the new line ###
# done
#
# &&
# echo "==================== CartPole =======================" ####
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -dsa --exp_name q1_sb_no_rtg_dsa --seed 1 5 10 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -rtg -dsa --exp_name q1_sb_rtg_dsa  --seed 1 5 10  &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -rtg --exp_name q1_sb_rtg_na --seed 1 5 10 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -dsa --exp_name q1_lb_no_rtg_dsa  --seed 1 5 10 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -rtg -dsa --exp_name q1_lb_rtg_dsa --seed 1 10 &&
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -rtg --exp_name q1_lb_rtg_na --seed 1 10 &&
