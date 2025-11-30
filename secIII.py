from utils import *
import time

# region setup
accel_filepath = 'project/projectfiles/secIII_acc.csv'
gyro_filepath = 'project/projectfiles/secIII_gyr.csv'
gps_filepath = 'project/projectfiles/secIII_gps.csv'

accel, gyro, gps = get_data_sec_III(accel_filepath, gyro_filepath, gps_filepath)

t_acc, a_acc = accel[:,0], accel[:,2]
t_gyr, w_gyr = gyro[:,0], gyro[:,3]
t_gps, lat_gps, long_gps = gps[:,0], gps[:,1], gps[:,2]
px_gps, py_gps = convert_gps(lat_gps, long_gps)

offset = t_gyr[0]-t_acc[0]
t_sync = t_gyr - offset
# endregion setup

# region system setup

# gps measurement noise
R = np.array([[0.06, 0],
              [0, 0.06]])

# imu process noise
cov_nu_a = np.array([[12.5, 0],
                     [0, 0.001]])

# initial state
x_0 = np.array([[0],
                [0],
                [0],
                [83.3*np.pi/180]])

# initial state covariance
P_0 = np.eye(4)

def f_c(x, u):
    """
    system dynamics given in problem description
    """
    p_x, p_y, v, theta = x.ravel()
    a, w = u.ravel()

    return np.array([[v*np.cos(theta)],
                     [v*np.sin(theta)],
                     [a],
                     [w]])

def f_d(x, u, T):
    """
    forward-euler discretization of f(x,u)
    x{k+1} = x{k} + x_dot{k}*T
    """
    p_xk, p_yk, vk, thetak = x.ravel()
    ak, wk = u.ravel()

    p_xk1 = p_xk + vk*np.cos(thetak)*T
    p_yk1 = p_yk + vk*np.sin(thetak)*T
    vk1 = vk + ak*T
    thetak1 = thetak + wk*T

    return np.array([[p_xk1],
                     [p_yk1],
                     [vk1],
                     [thetak1]])

def A_d(x, T):
    """
    forward-euler discretization of linearized A matrix
    A_d = I + A_c*T
    """

    p_x, p_y, v, theta = x.ravel()
    return np.array([[1, 0, T*np.cos(theta), -1*T*v*np.sin(theta)],
                    [0, 1, T*np.sin(theta), T*v*np.cos(theta)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])


# endregion system setup

# region EKF setup

def Q_k(T):
    """
    usually looks something like:
    [0  0   0       0
    0   0   0       0
    0   0   var*T^2 0
    0   0   0       var*T^2]
    """
    B = np.array([[0, 0],
                  [0, 0],
                  [T, 0],
                  [0, T]])
    
    return B @ cov_nu_a @ B.T

def pred(x_kk, u_k, P_kk, T):
    # predict next state
    x_k1k = f_d(x_kk, u_k, T)
    # get discrete A matrix from state and input
    A = A_d(x_kk, T)
    # get process noise covariance (IMU)
    Q = Q_k(T)
    # predict next state covariance
    P_k1k = A @ P_kk @ A.T + Q
    return x_k1k, P_k1k

def corr(P_k1k, R, x_k1k, y_k):
    # y~ = y_measured - y_predicted
    y_tilde = y_k - H @ x_k1k
    # output covariance
    K_yy = H @ P_k1k @ H.T + R
    K_k = P_k1k @ H.T @ np.linalg.inv(K_yy)
    # update state
    x_k1k1 = x_k1k + K_k @ y_tilde
    # update covariance
    P_k1k1 = P_k1k @ (np.eye(4) - H.T @ np.linalg.inv(K_yy) @ H @ P_k1k)

    return x_k1k1, P_k1k1

# endregion EKF setup


#--------------------------------EKF--------------------------------#
x = x_0
P = P_0

i = 0
tol = 0.3

xs = [x_0]

for k in range(0, len(t_sync)-1):
    #prediction step
    T = t_sync[k+1] - t_sync[k]

    u = np.array([[a_acc[k]],
                  [w_gyr[k]]])

    x_pred, P_pred = pred(x, u, P, T)

    #correction step (conditional on if there is a GPS measurement at a near moment)
    if i < len(t_gps) and np.abs(t_gps[i] - t_sync[k+1]) < tol:
        y = np.array([[px_gps[i]],
                      [py_gps[i]]])
        
        x, P = corr(P_pred, R, x_pred, y)
        i += 1
    else:
        x, P = x_pred, P_pred

    xs.append(x.copy())

xs = np.array(xs)
theta_ins = forward_euler_2(t_gyr, w_gyr)
# plot_2_data(t_sync, xs[:,3],t_gyr, theta_ins+83.3*np.pi/180, label_1='EKF', label_2='INS', y_axis='Heading Angle (rad)',title='Vehicle Heading')
# print(np.mean(a_acc), np.var(a_acc))
a_acc -= 2*0.016441781805342132

v_acc = forward_euler_2(t_sync, a_acc)
print(np.mean(a_acc), np.var(a_acc))


# dummy v value found using GPS measurements
# v_acc = np.mean(gps[1:,4])*np.ones(len(t_sync))

# plot_2_data(t_sync, xs[:,2],t_acc, v_acc, label_1='EKF', label_2='INS', y_axis='Speed (m/s)',title='Vehicle Speed')

#------------------------------INS, GPS, Kalman Position------------------------------#
x_acc = forward_euler_2(t_sync, v_acc*np.cos(theta_ins+83.3*np.pi/180))
y_acc = forward_euler_2(t_sync, v_acc*np.sin(theta_ins+83.3*np.pi/180))


plot_3_data(t_sync, xs[:,2], t_gps, gps[:,4], t_sync, v_acc)

# plot_3_data(xs[:,0], xs[:,1], px_gps, py_gps, x_acc, y_acc)


# 1. map v_acc and th_gyr to same timesteps
# 2. use vx = v*cos(theta) **OR SIN(THETA) (IDK?)
# 3. then integrate









