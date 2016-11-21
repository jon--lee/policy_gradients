import gym
import numpy as np
import IPython
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import tensorflow as tf
import matplotlib.pyplot as plt


def weight_variable(shape, stddev=.01):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, stddev=.01):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)


with tf.Graph().as_default():
    env = normalize(CartpoleEnv())

    N = 40
    SAMPLES = 50
    ITER = 100
    discount = 0.99
    learning_rate = .01

    state_dim   = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    sigma_squared = 1.0
    sigma = np.diag(np.ones(num_actions) * sigma_squared)

    observations_var = tf.placeholder(tf.float32, shape=(None, state_dim))
    actions_var = tf.placeholder(tf.float32, shape=(None, num_actions))
    returns_var = tf.placeholder(tf.float32, shape=(None))
    sigmas_var = tf.placeholder(tf.float32, shape=(None, num_actions))

    W1 = weight_variable([state_dim, 20])
    b1 = bias_variable([20])

    h1 = tf.nn.tanh(tf.matmul(observations_var, W1) + b1)

    W2 = weight_variable([20, num_actions])
    b2 = bias_variable([num_actions])

    outputs = tf.matmul(h1, W2) + b2

    dist = tf.contrib.distributions.MultivariateNormalDiag(outputs, sigmas_var)
    prob = dist.prob(actions_var)

    logprob = tf.log(prob)

    surr = -tf.reduce_mean(logprob * returns_var)

    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(surr)

    

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    def sample_control(observation):
        d = {observations_var: [observation]}
        with sess.as_default():
            control = outputs.eval(d)[0]
        random_control = np.random.multivariate_normal(control, sigma)
        return random_control

    with sess.as_default():
        total_rewards = []
        for i in range(ITER):
            paths = []
            sampleRewards = 0.0
            for n in range(N):
                observations = []
                actions = []
                rewards = []
                observation = env.reset()
                for samp in range(SAMPLES):
                    # env.render()
                    action = sample_control(observation)                    

                    next_observation, reward, terminal, _ = env.step(action)
                    observations.append(observation)
                    actions.append(action)
                    rewards.append(reward)
                    observation = next_observation
                    if terminal:
                        # Finish rollout if terminal state reached
                        break
                    sampleRewards += reward / (SAMPLES * N)

                returns = []
                return_so_far = 0
                for t in range(len(rewards) - 1, -1, -1):
                    return_so_far = rewards[t] + discount * return_so_far
                    returns.append(return_so_far)
                # The returns are stored backwards in time, so we need to revert it
                returns = returns[::-1]

                paths.append(dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                returns=np.array(returns),
                    ))

            observations = np.concatenate([p["observations"] for p in paths])
            actions = np.concatenate([p["actions"] for p in paths])
            returns = np.concatenate([p["returns"] for p in paths])
            sigmas = np.ones((len(observations), 1)) * sigma_squared


            sampleRewards = sampleRewards
            sess.run(opt, feed_dict = {observations_var: observations, actions_var: actions, returns_var: returns, sigmas_var: sigmas})
            print("Iteration " + str(i) + " rewards: " + str(sampleRewards))
