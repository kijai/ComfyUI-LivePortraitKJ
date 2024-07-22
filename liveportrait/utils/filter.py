import torch
import numpy as np
from pykalman import KalmanFilter


def smooth(x_d_lst, shape, device, observation_variance=3e-6, process_variance=1e-5):
    # Reshape x_d_lst, skipping None values
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst if x is not None]

    if not x_d_lst_reshape:  # Check if x_d_lst_reshape is empty after filtering
        return [None] * len(x_d_lst)  # Return a list of Nones with the same length as x_d_lst

    x_d_stacked = np.vstack(x_d_lst_reshape)

    kf = KalmanFilter(
        initial_state_mean=x_d_stacked[0],
        n_dim_obs=x_d_stacked.shape[1],
        transition_covariance=process_variance * np.eye(x_d_stacked.shape[1]),
        observation_covariance=observation_variance * np.eye(x_d_stacked.shape[1])
    )

    smoothed_state_means, _ = kf.smooth(x_d_stacked)

    # Initialize an iterator for smoothed_state_means
    smoothed_states_iter = iter(smoothed_state_means)

    # Create x_d_lst_smooth, inserting None for each None encountered in the original list
    x_d_lst_smooth = [torch.tensor(next(smoothed_states_iter).reshape(shape[-2:]), dtype=torch.float32, device=device) if x is not None else None for x in x_d_lst]

    return x_d_lst_smooth