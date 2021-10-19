import gym
env = gym.make('CartPole-v0')
from dqn import DQN
# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Hyper parameters
n_hidden = 50
batch_size = 32
lr = 0.01                 # learning rate
epsilon = 0.1             # epsilon-greedy
gamma = 0.9               # reward discount factor
target_replace_iter = 100 # target network 更新間隔
memory_capacity = 2000
n_episodes = 4000

# 建立 DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

# 學習
for i_episode in range(n_episodes):
    t = 0
    rewards = 0
    state = env.reset()
    while True:
        env.render()

        # 選擇 action
        action = dqn.choose_action(state)
        next_state, reward, done, info = env.step(action)
        # # 修改 reward，加快訓練
        # x, v, theta, omega = next_state
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 # 小車離中間越近越好
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # 柱子越正越好
        # reward = r1 + r2
        # 儲存 experience
        dqn.store_transition(state, action, reward, next_state)

        # 累積 reward
        rewards += reward

        # 有足夠 experience 後進行訓練
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        # 進入下一 state
        state = next_state

        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
            break

        t += 1

env.close()