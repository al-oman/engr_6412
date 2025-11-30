from utils import *

#-------------------load sensor data-------------------#
gyro1_filepath = 'project/projectfiles/secII_gyr_1.csv'
gyro2_filepath = 'project/projectfiles/secII_gyr_2.csv'
acc1_filepath = 'project/projectfiles/secII_acc_1.csv'
acc2_filepath = 'project/projectfiles/secII_acc_2.csv'

acc1_measured = read_csv(acc1_filepath, maxrows=15000)
acc2_measured = read_csv(acc2_filepath, maxrows=15000)
gyro1_measured = read_csv(gyro1_filepath, maxrows=15000)
gyro2_measured = read_csv(gyro2_filepath, maxrows=15000)

#-------------------remove gravity-------------------#
acc1_measured[:,3] = acc1_measured[:,3] - 9.805
acc2_measured[:,3] = acc2_measured[:,3] - 9.805

#-------------------betas from secI-------------------#
beta_x = np.array([-2.35277828e-02, 2.43452574e-06])
beta_y = np.array([1.70842370e-01, -2.90981962e-06])
beta_z = np.array([-6.01435148e-02,  1.59776676e-06])

beta_i = np.array([ 2.64028817e-04, -1.06202825e-07])
beta_j = np.array([3.04772890e-04, 2.53658675e-08])
beta_k = np.array([-1.26759705e-04,  -9.20066629e-08])

#--------------make matrices of [1, t] for each point----------------#
X_a1 = np.column_stack((np.ones(acc1_measured.shape[0]), acc1_measured[:,0]))
X_a2 = np.column_stack((np.ones(acc2_measured.shape[0]), acc2_measured[:,0]))
X_g1 = np.column_stack((np.ones(gyro1_measured.shape[0]), gyro1_measured[:,0]))
X_g2 = np.column_stack((np.ones(gyro2_measured.shape[0]), gyro2_measured[:,0]))

acc1 = acc1_measured.copy()
acc2 = acc2_measured.copy()
gyro1 = gyro1_measured.copy()
gyro2 = gyro2_measured.copy()

#-------------------betas from secII-------------------#
# beta_x = beta_x = np.linalg.inv((X_a1.T @ X_a1)) @ X_a1.T @ acc1[:,1]
# bias_x = X_a1 @ beta_x
# beta_y = beta_y = np.linalg.inv((X_a1.T @ X_a1)) @ X_a1.T @ acc1[:,2]
# bias_y = X_a1 @ beta_y
# beta_z = beta_z = np.linalg.inv((X_a1.T @ X_a1)) @ X_a1.T @ acc1[:,3]
# bias_z = X_a1 @ beta_z
#-------------------subtract biases from measurements-------------------#
acc1[:,1:] -= np.column_stack((X_a1 @ beta_x, X_a1 @ beta_y, X_a1 @ beta_z))
acc2[:,1:] -= np.column_stack((X_a2 @ beta_x, X_a2 @ beta_y, X_a2 @ beta_z))
gyro1[:,1:] -= np.column_stack((X_g1 @ beta_i, X_g1 @ beta_j, X_g1 @ beta_k))
gyro2[:,1:] -= np.column_stack((X_g2 @ beta_i, X_g2 @ beta_j, X_g2 @ beta_k))


#-------------------integrate-------------------#
vel1 = forward_euler(acc1)
vel2 = forward_euler(acc2)

pos1 = forward_euler(vel1)
pos2 = forward_euler(vel2)

the1 = forward_euler(gyro1)
the2 = forward_euler(gyro2)

plot_data(np.column_stack((the2[:,0], the2[:,3])), x_var='', title='Vehicle 2 z-Axis Angle', y_axis='Angle (rad)')

# from scipy.integrate import cumulative_trapezoid
# t_a1 = acc1[:,0]
# z_a1 = acc1[:,3]

# z_v1 = cumulative_trapezoid(z_a1, t_a1, initial=0)
# z_p1 = cumulative_trapezoid(z_v1, t_a1, initial=0)

# plot_data(np.column_stack((t_a1, z_p1)))



