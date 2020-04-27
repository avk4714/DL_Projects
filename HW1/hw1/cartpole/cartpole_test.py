## CSE 571 - HW Submission || Aman V Kalia - 1327251

import numpy as np

# Global variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 25
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(15448) #12345

# State representation
# dtheta, dx, theta, x

kernel_length_scales = np.array([[240.507, 242.9594, 218.0256, 203.0197],
                                 [175.9314, 176.8396, 178.0185, 33.0219],
                                 [7.4687, 7.3903, 13.0914, 34.6307],
                                 [0.8433, 1.0499, 1.2963, 2.3903],
                                 [0.781, 0.9858, 1.7216, 31.2894],
                                 [23.1603, 24.6355, 49.9782, 219.185]])
kernel_scale_factors = np.array([3.5236, 1.3658, 0.7204, 1.1478])
noise_sigmas = np.array([0.0431, 0.0165, 0.0145, 0.0143])


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

# Squared exponential kernel calculation function definition
def squared_exponential_kernel_t(x_i, x_j, l, sigma_f):
    # import pdb; pdb.set_trace()
    n1 = x_i.shape[0]   # M dimension of training: [M-by-6]
    n2 = x_j.shape[0]   # H dimension of state: [H-by-6]
    """
    : param l: is a [M-by-1] vector of lengths for the 5 states and 1 action element
    """
    if n1 == n2:
        kernel = np.zeros((n1,n1,6))
        dist = np.zeros((n1,n1,6))
        t_kernel = np.ones((n1,n1))
    else:
        kernel = np.zeros((n1,n2,6))
        dist = np.zeros((n1,n2,6))
        t_kernel = np.ones((n1,n2))

    # Kernel for all the states and actions
    dist[:,:,0] = np.subtract.outer(x_i[:,0],x_j[:,0])
    kernel[:,:,0] = np.square(sigma_f) * np.exp(-0.5 * np.square(dist[:,:,0]) * np.power(l[0],-2.0))

    dist[:,:,1] = np.subtract.outer(x_i[:,1],x_j[:,1])
    kernel[:,:,1] = np.square(sigma_f) * np.exp(-0.5 * np.square(dist[:,:,1]) * np.power(l[1],-2.0))

    dist[:,:,2] = np.subtract.outer(x_i[:,2],x_j[:,2])
    kernel[:,:,2] = np.square(sigma_f) * np.exp(-0.5 * np.square(dist[:,:,2]) * np.power(l[2],-2.0))

    dist[:,:,3] = np.subtract.outer(x_i[:,3],x_j[:,3])
    kernel[:,:,3] = np.square(sigma_f) * np.exp(-0.5 * np.square(dist[:,:,3]) * np.power(l[3],-2.0))

    dist[:,:,4] = np.subtract.outer(x_i[:,4],x_j[:,4])
    kernel[:,:,4] = np.square(sigma_f) * np.exp(-0.5 * np.square(dist[:,:,4]) * np.power(l[4],-2.0))

    dist[:,:,5] = np.subtract.outer(x_i[:,5],x_j[:,5])
    kernel[:,:,5] = np.square(sigma_f) * np.exp(-0.5 * np.square(dist[:,:,5]) * np.power(l[5],-2.0))

    # Combining the kernel
    t_kernel = kernel[:,:,0] * kernel[:,:,1] * kernel[:,:,2] * kernel[:,:,3] * kernel[:,:,4] * kernel[:,:,5]

    return t_kernel

def predict_squared_exponential_kernel_t(train_x_t, train_y_t, test_x_t, l_x, sigma_f_x, noise_sigma_x):
    """
    :param train_x_t: a numpy array of size [1 x 6]
    :param train_y_t: float parameter.
    :param test_x_t:  a numpy array of size [1 x 6]
    :param l_x: a numpy array of size [6 x 1]. length parameter of kernel. float
    :param sigma_f_x: scale parameter of kernel. float
    :param noise_sigma: noise standard deviation. float

    :return: mean: a numpy array of size [M]
             variance: a numpy array of size [M]

        Note: only return the variances, not the covariances
              i.e. the diagonal of the covariance matrix
    """

    # Pre-processing
    M = train_x_t.shape[0]
    # import pdb; pdb.set_trace()
    if test_x_t.ndim == 2:
    	H = test_x_t.shape[0]
    else:
    	H = test_x_t.shape[1]

    mean = np.zeros((1,1))
    variance = np.zeros((1,1))

    # Step 1: Calculate p_var = K + sigma_n^2*I
    K = np.zeros((M,M))
    K = squared_exponential_kernel_t(train_x_t, train_x_t, l_x, sigma_f_x)
    p_var = K + (np.square(noise_sigma_x) * np.eye(M))

    # Step 2: Compute p_var_inv
    p_var_inv = np.linalg.inv(p_var)

    # Step 3: Calculate k_s and k_sT using test data
    k_s = np.zeros((M,H))
    k_s = squared_exponential_kernel_t(train_x_t, test_x_t, l_x, sigma_f_x)

    # Step 4: Calculate mean = k_sT.p_var_inv.train_y
    mean = k_s.T.dot(p_var_inv).dot(train_y_t)

    # Step 5: Calculate k_ss
    k_ss = np.zeros((H,H))
    k_ss = squared_exponential_kernel_t(test_x_t, test_x_t, l_x ,sigma_f_x)
    tmpvar = k_ss - k_s.T.dot(p_var_inv).dot(k_s)
    variance = tmpvar.diagonal()

    return mean, variance

def predict_gp(train_x, train_y, init_state, action_traj):
    """
    Let M be the number of training examples
    Let H be the length of an epoch (NUM_DATAPOINTS_PER_EPOCH)
    Let N be the number of trajectories (NUM_TRAJ_SAMPLES)

    NOTE: Please use rng.normal(mu, sigma) to generate Gaussian random noise.
          https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.normal.html


    :param train_x: a numpy array of size [M x 6]
    :param train_y: a numpy array of size [M x 4]
    :param init_state: a numpy array of size [4]. Initial state of current epoch.
                       Use this to generate rollouts.
    :param action_traj: a numpy array of size [M]

    :return:
             # This is the mean rollout
             pred_gp_mean: a numpy array of size [H x 4]
                           This is mu_t[k] in Algorithm 1 in the HW1 PDF.
             pred_gp_variance: a numpy array of size [H x 4].
                               This is sigma_t[k] in Algorithm 1 in the HW1 PDF.
             rollout_gp: a numpy array of size [H x 4]
                         This is x_t[k] in Algorithm 1 in the HW1 PDF.
                         It should start from t=1, i.e. rollout_gp[0,k] = x_1[k]

             # These are the sampled rollouts
             pred_gp_mean_trajs: a numpy array of size [N x H x 4]
                                 This is mu_t^j[k] in Algorithm 2 in the HW1 PDF.
             pred_gp_variance_trajs: a numpy array of size [N x H x 4]
                                     This is sigma_t^j[k] in Algorithm 2 in the HW1 PDF.
             rollout_gp_trajs: a numpy array of size [N x H x 4]
                               This is x_t^j[k] in Algorithm 2 in the HW1 PDF.
                               It should start from t=1, i.e. rollout_gp_trajs[j,0,k] = x_1^j[k]
    """

    # TODO: Compute these variables.
    pred_gp_mean = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))

    pred_gp_mean_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))

    # Training data extraction
    train_x_t = np.copy(train_x)
    train_y_t = np.copy(train_y)
    # M = train_x_t.shape[0]
    # import pdb; pdb.set_trace()
    ## Mean Rollout GP - Implementation
    aug_x = np.zeros((NUM_DATAPOINTS_PER_EPOCH,6))
    # import pdb; pdb.set_trace()

    state_x = np.copy(init_state)
    for i in range(NUM_DATAPOINTS_PER_EPOCH):
        # Step 1: Augment state
        aug_x[i,:] = augmented_state(state_x,action_traj[i])
        # import pdb; pdb.set_trace()

        # Step 2a: State 1 training
        pred_gp_mean[i,0], pred_gp_variance[i,0] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,0], aug_x, kernel_length_scales[:,0].reshape((6,1)), kernel_scale_factors[0], noise_sigmas[0]))[:,i]

        # Step 2b: State 2 training
        pred_gp_mean[i,1], pred_gp_variance[i,1] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,1], aug_x, kernel_length_scales[:,1].reshape((6,1)), kernel_scale_factors[1], noise_sigmas[1]))[:,i]

        # Step 2c: State 3 training
        pred_gp_mean[i,2], pred_gp_variance[i,2] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,2], aug_x, kernel_length_scales[:,2].reshape((6,1)), kernel_scale_factors[2], noise_sigmas[2]))[:,i]

        # Step 2d: State 4 training
        pred_gp_mean[i,3], pred_gp_variance[i,3] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,3], aug_x, kernel_length_scales[:,3].reshape((6,1)), kernel_scale_factors[3], noise_sigmas[3]))[:,i]

        # Step 3: State update
        state_x += pred_gp_mean[i,:]
        rollout_gp[i,:] = np.copy(state_x)

    ## Sampled Rollout GP
    state_traj_x = np.zeros((NUM_TRAJ_SAMPLES,4))
    aug_traj_x = np.zeros((NUM_TRAJ_SAMPLES,NUM_DATAPOINTS_PER_EPOCH,6))
    # vect_aug_state = np.vectorize(augmented_state)
    # for j in range(NUM_TRAJ_SAMPLES):
    state_traj_x[:,:] = np.copy(init_state)
    # import pdb; pdb.set_trace()
    for k in range(NUM_DATAPOINTS_PER_EPOCH):
        # Step 1: Augment State
        aug_traj_x[:,k,:] = np.array([augmented_state(state_traj_xi,action_traj[k]) for state_traj_xi in state_traj_x[:,:]])
        # import pdb; pdb.set_trace()
        # Step 2a: State 1 training
        pred_gp_mean_trajs[:,k,0], pred_gp_variance_trajs[:,k,0] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,0], aug_traj_x[:,k,:], kernel_length_scales[:,0].reshape((6,1)), kernel_scale_factors[0], noise_sigmas[0]))

        # Step 2b: State 2 training
        pred_gp_mean_trajs[:,k,1], pred_gp_variance_trajs[:,k,1] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,1], aug_traj_x[:,k,:], kernel_length_scales[:,1].reshape((6,1)), kernel_scale_factors[1], noise_sigmas[1]))

        # Step 2c: State 3 training
        pred_gp_mean_trajs[:,k,2], pred_gp_variance_trajs[:,k,2] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,2], aug_traj_x[:,k,:], kernel_length_scales[:,2].reshape((6,1)), kernel_scale_factors[2], noise_sigmas[2]))

        # Step 2d: State 4 training
        pred_gp_mean_trajs[:,k,3], pred_gp_variance_trajs[:,k,3] = np.asarray(predict_squared_exponential_kernel_t(
        train_x_t, train_y_t[:,3], aug_traj_x[:,k,:], kernel_length_scales[:,3].reshape((6,1)), kernel_scale_factors[3], noise_sigmas[3]))

        # Step 3: Gaussian Sampling
        sample_gp = rng.normal(pred_gp_mean_trajs[:,k,:],np.sqrt(np.abs(pred_gp_variance_trajs[:,k,:])))
        # Step 4: State Update
        
        state_traj_x += sample_gp
        # import pdb; pdb.set_trace()
        rollout_gp_trajs[:,k,:] = np.copy(state_traj_x)
        # import pdb; pdb.set_trace()

    return pred_gp_mean, pred_gp_variance, rollout_gp, pred_gp_mean_trajs, pred_gp_variance_trajs, rollout_gp_trajs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from visualization import Visualizer
    import cv2

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    # Initial training data used to train GP for the first epoch
    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
    delta_state_traj = state_traj[1:] - state_traj[:-1]
    train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)

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

        (pred_gp_mean,
         pred_gp_variance,
         rollout_gp,
         pred_gp_mean_trajs,
         pred_gp_variance_trajs,
         rollout_gp_trajs) = predict_gp(train_x, train_y, state_traj[0], action_traj)

        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

            if i == 0:
                vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
                vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [state_traj[i][2]] * NUM_TRAJ_SAMPLES)
            else:
                vis.set_gp_cartpole_state(rollout_gp[i-1][3], rollout_gp[i-1][2])
                vis.set_gp_cartpole_rollout_state(rollout_gp_trajs[:, i-1, 3], rollout_gp_trajs[:, i-1, 2])

            vis.set_gp_delta_state_trajectory(ts[:i+1], pred_gp_mean[:i+1], pred_gp_variance[:i+1])

            if policy == swingup_policy:
                policy_type = 'swing up'
            else:
                policy_type = 'random'

            vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i==0))
            cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('cartpole.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))

        # Augment training data
        new_train_x, new_train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        train_x = np.concatenate([train_x, new_train_x])
        train_y = np.concatenate([train_y, new_train_y])
