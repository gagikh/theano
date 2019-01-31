import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

"""
Parameters:    
x: initial state 6-tuple of location and velocity: (x, y, z, x_dot, y_dot, z_dot)
P: initial uncertainty convariance matrix
measurement: observed position
R: measurement noise 
motion: external motion added to state vector x
Q: motion noise (same shape as P)
F: next state function: x_prime = F*x
H: measurement function: position = H*x
"""
def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

def demo_kalman_xyz(observed_x, observed_y, observed_z):

    N = 3
    # x: initial state 2N-tuple of location and velocity: (x0, x1, ..xn, x0_dot, x1_dot, .. xn_dot)
    num_states = 2 * N
    ugain = 100

    x = np.matrix([0] * num_states).T
    P = np.matrix(np.eye(num_states)) * ugain # initial uncertainty
    M = np.matrix([0] * num_states).T
    Q = np.matrix(np.eye(num_states))

    F = np.matrix(np.eye(num_states))
    F[0, 3] = 1
    F[1, 4] = 1
    F[2, 5] = 1
    H = np.matrix(np.eye(N, num_states))

    result = []
    R = 0.01**2
    for meas in zip(observed_x, observed_y, observed_z):
        x, P = kalman(x, P, meas, R, M, Q, F, H)
        print x
        result.append((x[:N]).tolist())

    kalman_x, kalman_y, kalman_z = zip(*result)
    out_x = np.asarray(kalman_x).reshape(-1)
    out_y = np.asarray(kalman_y).reshape(-1)
    out_z = np.asarray(kalman_z).reshape(-1)

    return out_x, out_y, out_z

def main():
    num_points = 100

    true_x = np.linspace(0.0, 10.0, num_points)
    true_y = 5 + true_x**2
    true_z = 6 + 3*true_x + 0.05 * true_x**3

    observed_x = true_x + 0.05*np.random.random(num_points)*true_x
    observed_y = true_y + 0.05*np.random.random(num_points)*true_y
    observed_z = true_z + 0.05*np.random.random(num_points)*true_z

    out_x, out_y, out_z = demo_kalman_xyz(observed_x, observed_y, observed_z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(observed_x, observed_y, observed_z, 'r-')

    ax.plot(out_x, out_y, out_z, 'g-')
    plt.show()

main()
