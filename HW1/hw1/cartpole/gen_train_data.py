# Run this script to generate training data for the cartpole dynamics learning.
import numpy as np
import random

# Global Variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
# rng = np.random.RandomState(54454) #12345

# Function Definitions
def sim_rollout(sim, policy, n_steps, dt, init_state):
    """
    :param sim: the simulator
    :param policy: policy that generates rollout
    :param n_steps: number of time steps to run
    :param dt: simulation step size
    :param init_state: initial state

    :return: times:   a numpy array of size [n_steps + 1]
             states:  a numpy array of size [n_steps + 1 x 4]
             actions: a numpy array of size [n_steps]
                        actions[i] is applied to states[i] to generate states[i+1]
    """
    states = []
    state = init_state
    actions = []
    for i in range(n_steps):
        states.append(state)
        action = policy.predict(state)
        actions.append(action)
        state = sim.step(state, [action], noisy=True)

    states.append(state)
    times = np.arange(n_steps + 1) * dt
    return times, np.array(states), np.array(actions)

def augmented_state(state, action):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    dtheta, dx, theta, x = state
    return x, dx, dtheta, np.sin(theta), np.cos(theta), action


def make_training_data(state_traj, action_traj, delta_state_traj):
    """
    A helper function to generate training data.
    """
    x = np.array([augmented_state(state, action) for state, action in zip(state_traj, action_traj)])
    y = delta_state_traj
    return x, y


# Main Program
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from visualization import Visualizer
    import cv2
    import csv

    # Getting policy for dynamics and initializing simulator
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    sim = CartpoleSim(dt=DELTA_T)

    '''
    Number of training sets of NUM_DATAPOINTS_PER_EPOCH each to generate. You
    can change this value to generate NUM_TRAIN_DATASETS * NUM_DATAPOINTS_PER_EPOCH
    sized datasets.
    '''
    NUM_TRAIN_DATASETS = 100

    # Loop to simulate the cartpole dynamics with different initial state
    # conditions.
    for i in range(NUM_TRAIN_DATASETS):
        random_policy = RandomPolicy(seed=np.random.randint(12121,64923))
        rng = np.random.RandomState(np.random.randint(12121,64923))
        # Setting for random policy to generate training data
        # Every 5th dataset is SwingUp Policy
        if (i + 1) % 5 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
        ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]
        train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        if i == 0:
            train_x_full = train_x
            train_y_full = train_y
        else:
            train_x_full = np.append(train_x_full,train_x,axis=0)
            train_y_full = np.append(train_y_full,train_y,axis=0)

    # Writing data to CSV
    with open('train_x_data_2.csv', mode='w') as train_x_dat:
        train_x_writer = csv.writer(train_x_dat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(NUM_TRAIN_DATASETS * NUM_DATAPOINTS_PER_EPOCH):
            train_x_writer.writerow(train_x_full[i,:])

    with open('train_y_data_2.csv', mode='w') as train_y_dat:
        train_y_writer = csv.writer(train_y_dat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(NUM_TRAIN_DATASETS * NUM_DATAPOINTS_PER_EPOCH):
            train_y_writer.writerow(train_y_full[i,:])
