from utils import *

gyro_filepath = 'project/projectfiles/secI_gyr.csv'
acc_filepath = 'project/projectfiles/secI_acc.csv'

"""
Part 1: Bias Calibration Using Linear Least Squares
-----------------------------------------------------------------------------

if bias is of the shape:
    y = beta_0 * 1 + beta+1 * t
then, using least squares:
    beta = (x.T * x)^-1 * x.T * y
*FIND CITATION
"""

#import accel data
measured_accel_data = read_csv(acc_filepath, maxrows=15000)
#remove gravity
measured_accel_data[:,3] = measured_accel_data[:,3] - 9.805
t_a = measured_accel_data[:,0]
x_measured = measured_accel_data[:,1]
y_measured = measured_accel_data[:,2]
z_measured = measured_accel_data[:,3]

plot_data(np.column_stack((measured_accel_data[:,0],z_measured)))

#dimensions
m = measured_accel_data.shape[0]
X = np.column_stack((np.ones(m), measured_accel_data[:,0]))

#find betas for each direction
beta_x = np.linalg.inv((X.T @ X)) @ X.T @ x_measured
beta_y = np.linalg.inv((X.T @ X)) @ X.T @ y_measured
beta_z = np.linalg.inv((X.T @ X)) @ X.T @ z_measured

#find predicted values
x_bias = X @ beta_x
y_bias = X @ beta_y
z_bias = X @ beta_z

# plot_data(np.column_stack((X[:,1], x_measured, x_bias)), x_var='Measured', y_var='Bias', title='X Acceleration', y_axis='Acceleration [m/s^2]')
# plot_data(np.column_stack((X[:,1], y_measured, y_bias)), x_var='Measured', y_var='Bias', title='Y Acceleration', y_axis='Acceleration [m/s^2]')
# plot_data(np.column_stack((X[:,1], z_measured, z_bias)), x_var='Measured', y_var='Bias', title='Z Acceleration', y_axis='Acceleration [m/s^2]')

#import gyro data
measured_gyro_data = read_csv(gyro_filepath, maxrows=15000)
i_measured = measured_gyro_data[:,1]
j_measured = measured_gyro_data[:,2]
k_measured = measured_gyro_data[:,3]
t_g = measured_gyro_data[:,0]

m = measured_gyro_data.shape[0]
X = np.column_stack((np.ones(m), measured_gyro_data[:,0]))

beta_i = np.linalg.inv((X.T @ X)) @ X.T @ i_measured
beta_j = np.linalg.inv((X.T @ X)) @ X.T @ j_measured
beta_k = np.linalg.inv((X.T @ X)) @ X.T @ k_measured

i_bias = X @ beta_i
j_bias = X @ beta_j
k_bias = X @ beta_k

# plot_data(np.column_stack((X[:,1], i_measured, i_bias)), x_var='Measured', y_var='Bias', title='X Gyro', y_axis='Angular Velocity [rad/s]')
# plot_data(np.column_stack((X[:,1], j_measured, j_bias)), x_var='Measured', y_var='Bias', title='Y Gyro', y_axis='Angular Velocity [rad/s]')
# plot_data(np.column_stack((X[:,1], k_measured, k_bias)), x_var='Measured', y_var='Bias', title='Z Gyro', y_axis='Angular Velocity [rad/s]')

"""
Part 2: Measurement Noise Calibration
-----------------------------------------------------------------------------

Cov(x,y) = E[(x-E[x])(y-E[y])^T] = E[xy^T] - E[x]E[y]

since:
    a_m(t) = a(t) + b_a(t) + v_a(t)
    w_m(t) = w(t) + b_w(t) + v_w(t)
    and IMU is stationary ie: a(t) = w(t) = 0
then:
    v_a(t) = a_m(t) - b_a(t)
    v_w(t) = w_m(t) - b_w(t)

"""


#measurement noise for acceleration and gyro
v_x = x_measured - x_bias
v_y = y_measured - y_bias
v_z = z_measured - z_bias
v_i = i_measured - i_bias
v_j = j_measured - j_bias
v_k = k_measured - k_bias

#variance, expected value, and covariance for accelerometer noise
var_x = np.mean(v_x**2) - np.mean(v_x)**2
var_y = np.mean(v_y**2) - np.mean(v_y)**2
var_z = np.mean(v_z**2) - np.mean(v_z)**2
mean_x = np.mean(v_x)
mean_y = np.mean(v_y)
mean_z = np.mean(v_z)
cov_xy = np.mean(v_x*v_y) - np.mean(v_x)*np.mean(v_y)
cov_xz = np.mean(v_x*v_z) - np.mean(v_x)*np.mean(v_z)
cov_yz = np.mean(v_y*v_z) - np.mean(v_y)*np.mean(v_z)

cov_accel = np.array([[var_x, cov_xy, cov_xz],
                      [cov_xy, var_y, cov_yz],
                      [cov_xz, cov_yz, var_z]])

#variance, expected value, and covariance for gyroscope noise
var_i = np.mean(v_i**2) - np.mean(v_i)**2
var_j = np.mean(v_j**2) - np.mean(v_j)**2
var_k = np.mean(v_k**2) - np.mean(v_k)**2
mean_i = np.mean(v_i)
mean_j = np.mean(v_j)
mean_k = np.mean(v_k)
cov_ij = np.mean(v_i*v_j) - np.mean(v_i)*np.mean(v_j)
cov_ik = np.mean(v_i*v_k) - np.mean(v_i)*np.mean(v_k)
cov_jk = np.mean(v_j*v_k) - np.mean(v_j)*np.mean(v_k)

cov_gyro = np.array([[var_i, cov_ij, cov_ik],
                     [cov_ij, var_j, cov_jk],
                     [cov_ik, cov_jk, var_k]])

plot_data(np.column_stack((measured_accel_data[:,0], z_measured, z_bias )), x_var='Measured', y_var='Bias')

# histogram(v_i, mean=mean_i, var=var_i, bins=20, title='Histogram of Gyroscope X Noise')
# histogram(v_j, mean=mean_j, var=var_j, bins=20, title='Histogram of Gyroscope Y Noise')
# histogram(v_k, mean=mean_k, var=var_k, bins=12, title='Histogram of Gyroscope Z Noise')

# print(beta_x, beta_y, beta_z)
# print(beta_i, beta_j, beta_k)
