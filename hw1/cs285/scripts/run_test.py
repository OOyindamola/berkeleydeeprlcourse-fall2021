from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import numpy as np
import tensorflow as tf
import time



from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy

def save_frames_as_gif(frames,episode_num,  path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.text(0.6, 0.6, f'Episode: {episode_num+1}',
         fontsize=20)
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


loaded_expert_policy = LoadedGaussianPolicy('/home/oyindamola/Research/homework_fall2021/hw1/cs285/policies/experts/Ant.pkl')
sess = tf.compat.v1.Session()
env = gym.make('Ant-v2')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print (obs_dim,act_dim)

for i in range(500):
    obs = env.reset()
    frames = []
    for t in range(500):
        # env.render(mode='human')
        frames.append(env.render(mode="rgb_array"))
        # action = np.random.randn(act_dim,1)
        # action = action.reshape((1,-1)).astype(np.float32)
        action =  loaded_expert_policy.get_action(obs)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        # print(env.model.opt.timestep)
        # time.sleep(.1)
        if done:
            break
        # time.sleep(0.01*env.model.opt.timestep)
    env.close()
    save_frames_as_gif(frames,i)
