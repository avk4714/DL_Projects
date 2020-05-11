import numpy as np

# Global variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(54454) #12345

# State representation
# dtheta, dx, theta, x

# kernel_length_scales = np.array([[240.507, 242.9594, 218.0256, 203.0197],
#                                  [175.9314, 176.8396, 178.0185, 33.0219],
#                                  [7.4687, 7.3903, 13.0914, 34.6307],
#                                  [0.8433, 1.0499, 1.2963, 2.3903],
#                                  [0.781, 0.9858, 1.7216, 31.2894],
#                                  [23.1603, 24.6355, 49.9782, 219.185]])
# kernel_scale_factors = np.array([3.5236, 1.3658, 0.7204, 1.1478])
# noise_sigmas = np.array([0.0431, 0.0165, 0.0145, 0.0143])


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
    from visualization import Visualizer
    from deep_ensembles_nn import Net
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

    # Initial training data used to train GP for the first epoch
    # init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    # ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
    # delta_state_traj = state_traj[1:] - state_traj[:-1]
    # train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)

    # Load DNN Model with the trained data
    n_hidden = 1000
    d_in = 6            # Inputs are: [p, dp, dtheta, sin(theta), cos(theta), action]
    d_out = 4*2           # Outputs are: [ddtheta, ddp, dtheta, dp]
    NUM_MODEL_ITERATIONS = 50

    model0 = Net(d_in, d_out, n_hidden)
    PATH0 = 'cartpole_deep_ensembles_0.pth'
    model0.load_state_dict(torch.load(PATH0))
    model0.eval()

    model1 = Net(d_in, d_out, n_hidden)
    PATH1 = 'cartpole_deep_ensembles_1.pth'
    model1.load_state_dict(torch.load(PATH1))
    model1.eval()

    model2 = Net(d_in, d_out, n_hidden)
    PATH2 = 'cartpole_deep_ensembles_2.pth'
    model2.load_state_dict(torch.load(PATH2))
    model2.eval()

    model3 = Net(d_in, d_out, n_hidden)
    PATH3 = 'cartpole_deep_ensembles_3.pth'
    model3.load_state_dict(torch.load(PATH3))
    model3.eval()

    model4 = Net(d_in, d_out, n_hidden)
    PATH4 = 'cartpole_deep_ensembles_4.pth'
    model4.load_state_dict(torch.load(PATH4))
    model4.eval()
    ensemble = [model0, model1, model2, model3, model4]

    for epoch in range(NUM_TRAINING_EPOCHS):
        vis.clear()

        # Use learned policy every 4th epoch
        if (epoch + 1) % 4 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]

        # Deep ensembles
        
        for j in range(NUM_DATAPOINTS_PER_EPOCH):
            if j == 0:
                pred_state_traj = init_state.reshape(1,4)
                pred_state = init_state.reshape(1,4)
            	# import pdb; pdb.set_trace()
            	# pred_state_traj = np.append(pred_state_traj,pred_state.reshape(1,4),axis=0)

            # import pdb; pdb.set_trace()
            aug_state = augmented_state(pred_state_traj[j,:], action_traj[j])
            # import pdb; pdb.set_trace()
            aug_state_tnsr = Variable(torch.from_numpy(aug_state))
            # import pdb; pdb.set_trace()

            mean_s = []
            var_s = []
            # Run through the model
            for i in range(len(ensemble)):
                model = ensemble[i]
                mean, var = model(aug_state_tnsr.float().reshape(1,6)) #reshape is imprtant here
                mean_s.append(mean)
                var_s.append(var)
            
            pred_state_delta = torch.stack(mean_s).mean(dim=0)
            varations = (torch.stack(var_s) + torch.stack(mean_s).pow(2)).mean(dim=0) - pred_state_delta.pow(2)

            # tot_pred_state_delta = tot_pred_state_delta.reshape(NUM_MODEL_ITERATIONS,4)
            # Calculate mean and variance of the predicted states
            # import pdb; pdb.set_trace()
            pred_state_delta = pred_state_delta.detach().numpy().reshape(1,4)
            # import pdb; pdb.set_trace()

            # State update
            pred_state = np.add(pred_state,pred_state_delta)

            pred_state_traj = np.append(pred_state_traj,pred_state.reshape(1,4),axis=0)

        dnn_state_traj = np.copy(pred_state_traj)
        dnn_delta_state_traj = dnn_state_traj[1:] - dnn_state_traj[:-1]
        # import pdb; pdb.set_trace()
        # Plot Training Y and Predicted Y
        # t_vec = np.arange(0.0,(NUM_DATAPOINTS_PER_EPOCH + 1) * DELTA_T,DELTA_T)
        # # import pdb; pdb.set_trace()
        # plt.plot(t_vec,state_traj[:,2])
        # plt.plot(t_vec[:-1],dnn_state_traj.T[:,2],'--')
        # plt.grid()
        # # plt.ylim((-10,10))
        # # plt.xlim((-10,10))
        # plt.savefig("test_comparison_3.png")
        # plt.show()
        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

            vis.set_dnn_cartpole_state(dnn_state_traj[i][3], dnn_state_traj[i][2])
            vis.set_dnn_delta_state_trajectory(ts[:i+1], dnn_delta_state_traj[:i+1])

            if policy == swingup_policy:
                policy_type = 'swing up'
            else:
                policy_type = 'random'

            vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i==0))
            # cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('deep_ensembles.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))

        # Augment training data
        # new_train_x, new_train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        # train_x = np.concatenate([train_x, new_train_x])
        # train_y = np.concatenate([train_y, new_train_y])
