'''Dropout Method Implementation for testing in PyTorch
By - Aman V. Kalia for CSE 571 Project'''

import numpy as np

# Global variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
DELTA_T = 0.05
rng = np.random.RandomState(54454) #12345

# State representation
# dtheta, dx, theta, x

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
    dtheta = state[0]
    dx = state[1]
    theta = state[2]
    x = state[3]


    return np.array([x, dx, dtheta, np.sin(theta), np.cos(theta), action])


def make_training_data(state_traj, action_traj, delta_state_traj):
    """
    A helper function to generate training data.
    """
    x = np.array([augmented_state(state, action) for state, action in zip(state_traj, action_traj)])
    y = delta_state_traj
    return x, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from dropout_visualization import Visualizer
    from cartpole_dnn import Net
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch.distributions import Bernoulli
    import matplotlib.pyplot as plt
    import csv

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    '''Load DNN Model with the trained data'''
    n_hidden = 1000
    z_prob = 0.2
    lam_mult = 1e-2
    d_in = 6            # Inputs are: [p, dp, dtheta, sin(theta), cos(theta), action]
    d_out = 4           # Outputs are: [ddtheta, ddp, dtheta, dp]
    NUM_MODEL_ITERATIONS = 50

    '''Trained model is loaded as two separate instances to eval
    and non-eval modes'''
    PATH = 'cartpole_ReLU_7.pth'
    model = Net(d_in, n_hidden, d_out, z_prob, lam_mult)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    mc_model = Net(d_in, n_hidden, d_out, z_prob, lam_mult)
    mc_model.load_state_dict(torch.load(PATH))

    for epoch in range(NUM_TRAINING_EPOCHS):
        vis.clear()

        # Use learned policy every 4th epoch
        if (epoch + 1) % 4 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy,
                                NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]

        '''DNN Rollout'''
        for j in range(NUM_DATAPOINTS_PER_EPOCH):
            if j == 0:
                pred_state_traj = init_state.reshape(1,4)
                pred_state = init_state.reshape(1,4)
                mu_pred_state_traj = []
                var_pred_state_traj = []

            aug_state = augmented_state(pred_state_traj[j,:], action_traj[j])
            aug_state_tnsr = Variable(torch.from_numpy(aug_state))

            '''Run through the model '''
            pred_state_delta = model(aug_state_tnsr.float()).detach().numpy()
            mc_pred_state_delta = []
            '''To generate different possible outcomes to calculate mean
            and variances'''
            for i in range(NUM_MODEL_ITERATIONS):
            	mc_pred_state_delta = np.append(mc_pred_state_delta,mc_model(aug_state_tnsr.float()).detach().numpy(),axis=0)

            mc_pred_state_delta = mc_pred_state_delta.reshape(NUM_MODEL_ITERATIONS,4)
            mu_mc_pred_state_delta = []
            var_mc_pred_state_delta = []
            for k in range(4):
            	mu_mc_pred_state_delta = np.append(mu_mc_pred_state_delta, np.mean(mc_pred_state_delta[:,k]))
            	var_mc_pred_state_delta = np.append(var_mc_pred_state_delta, np.var(mc_pred_state_delta[:,k]))

            pred_state_delta = pred_state_delta.reshape(1,4)

            '''State update'''
            pred_state = np.add(pred_state,pred_state_delta)

            pred_state_traj = np.append(pred_state_traj,pred_state.reshape(1,4),axis=0)
            mu_pred_state_traj = np.append(mu_pred_state_traj, mu_mc_pred_state_delta)
            var_pred_state_traj = np.append(var_pred_state_traj, var_mc_pred_state_delta)

        mu_dnn_state_traj = np.copy(mu_pred_state_traj)
        mu_dnn_state_traj = mu_dnn_state_traj.reshape(NUM_DATAPOINTS_PER_EPOCH,4)
        var_dnn_state_traj = np.copy(var_pred_state_traj)
        var_dnn_state_traj = var_dnn_state_traj.reshape(NUM_DATAPOINTS_PER_EPOCH,4)
        dnn_state_traj = np.copy(pred_state_traj)
        dnn_delta_state_traj = dnn_state_traj[1:] - dnn_state_traj[:-1]

        '''Visualization Section'''
        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])
            # import pdb; pdb.set_trace()
            vis.set_dnn_cartpole_state(dnn_state_traj[i][3], dnn_state_traj[i][2])
            vis.set_dnn_delta_state_trajectory(ts[:i+1], mu_dnn_state_traj[:i+1], var_dnn_state_traj[:i+1])
            # vis.set_dnn_delta_state_trajectory(ts[:i+1], dnn_delta_state_traj[:i+1])

            if policy == swingup_policy:
                policy_type = 'swing up'
            else:
                policy_type = 'random'

            vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i==0))
            cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('cartpole_dnn_7.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))
