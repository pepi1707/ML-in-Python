import gym
import numpy as np
import time

def QLearn(env_name = "MountainCar-v0", min_lr = 0.003, init_lr = 1.0,  gamma = 1.0, epochs = 10000,
           render = False, eps = 0.02):

    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    n_acts = env.action_space.n
    obs_shape = np.array([40] * env.observation_space.shape[0] + [n_acts])

    q_table = np.zeros((obs_shape.astype(np.int)))

    def compress(obs):
        comp = np.zeros_like(obs)
        for i in range(len(obs)):
            comp[i] = obs[i] - env.observation_space.low[i]
            comp[i] = comp[i] * obs_shape[i] / (env.observation_space.high[i] - env.observation_space.low[i])
            comp[i] = int(comp[i])
        return comp.astype(np.int)


    def train_one_epoch():

        obs = env.reset()

        done = False
        rewards = 0
        while not done:
            if(render):
                env.render()
            comp_obs = compress(obs)

            if np.random.uniform(0, 1) < eps:
                act = np.random.choice(env.action_space.n)

            else:
                logits = np.exp(q_table[comp_obs[0]][comp_obs[1]])
                probs = logits / np.sum(logits)
                act = np.random.choice(env.action_space.n, p = probs)
            
            new_obs, rew, done, _ = env.step(act)
            rewards += rew
            
            new_obs_comp = compress(new_obs)
            q_table[comp_obs[0]][comp_obs[1]][act] = (1 - eta) * q_table[comp_obs[0]][comp_obs[1]][act] + eta * (rew + gamma * np.max(q_table[new_obs_comp[0]][new_obs_comp[1]]))
            obs = new_obs.copy()
        return rewards

    for i in range(epochs):
        eta = max(min_lr, init_lr * (0.85 ** (i // 100)))
        rewards = train_one_epoch()
        render = False
        if rewards != -200:
            print("SUCCESS")
            render = True
        #print('Epoch: %3d \t Reward: %.3f'% (i + 1, rewards))
        

if __name__ == '__main__':
    QLearn()