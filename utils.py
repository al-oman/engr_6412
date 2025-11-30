import numpy as np
import matplotlib.pyplot as plt

beta_x = np.array([-2.35277828e-02, 2.43452574e-06])
beta_y = np.array([1.70842370e-01, -2.90981962e-06])
beta_z = np.array([-6.01435148e-02,  1.59776676e-06])

beta_i = np.array([ 2.64028817e-04, -1.06202825e-07])
beta_j = np.array([3.04772890e-04, 2.53658675e-08])
beta_k = np.array([-1.26759705e-04,  -9.20066629e-08])

def read_csv(filepath, maxrows=1000):
    """
    reads CSV file and returns numpy array
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, max_rows=maxrows)
    data = data[:, :4]
    return data

def read_gps_csv(filepath, maxrows=1000):
    """
    reads GPS CSV file and returns numpy array
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, max_rows=maxrows)
    data = data[:, :]
    return data

def forward_euler(accel_data):
    """
    integrates using forward euler
    """
    n = accel_data.shape[0]
    vel_data = np.zeros((n, 4))
    vel_data[:, 0] = accel_data[:, 0]

    for i in range(1, n):
        dt = accel_data[i, 0] - accel_data[i-1, 0]
        vel_data[i, 1:] = vel_data[i-1, 1:] + accel_data[i-1, 1:] * dt

    return vel_data

def forward_euler_2(t, y):
    """
    integrates using forward euler
    """
    n = len(y)
    int_y = np.zeros(n)

    for i in range(1, n):
        dt = t[i] - t[i-1]
        int_y[i] = int_y[i-1] + y[i-1] * dt

    return int_y
    

def RK4(data, x0, v0):
    n = data.shape[0]
    vel = np.zeros((n, 4))
    pos = np.zeros((n, 4))

    return 0

def plot_data(data, x_var='X', y_var='Y', z_var='Z', y_axis='Value', title='Sensor Data'):
    """
    Plots 3 columns vs leftmost column (time)
    """
    time = data[:, 0]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    labels = ['time', x_var, y_var, z_var]
    for i in range(1, data.shape[1]):
        plt.plot(time, data[:, i], label=labels[i])
    plt.xlabel('Time (s)')
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_2_data(x1, y1, x2=None, y2=None, label_1='y1', label_2='y2', y_axis='Value', title='Sensor Data'):
    """
    Plots 3 columns vs leftmost column (time)
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)

    plt.plot(x1, y1, label=label_1)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2, label=label_2)

    plt.xlabel('Time (s)')
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3_data(x1, y1, x2=None, y2=None, x3=None, y3=None, label_1='y1', label_2='y2', label_3='y3', y_axis='Value', title='Sensor Data'):
    """
    Plots 3 columns vs leftmost column (time)
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)

    plt.plot(x1, y1, label=label_1)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2, label=label_2)
    if x3 is not None and y3 is not None:
        plt.plot(x3, y3, label=label_3)

    plt.xlabel('Time (s)')
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)
    plt.show()

def histogram(data, mean, var, bins=10, xlabel='Noise Value', title='Histogram'):
    #adjust data shape and get limits
    data = data.flatten()
    min = data.min()
    max = data.max()
    min = mean - 6*np.sqrt(var)
    max = mean + 6*np.sqrt(var)
    bin_width = (max - min) / bins
    print(bin_width)
    #put data into bins of equal width
    counts, bin_edges = np.histogram(data, bins=bins, range=(min, max))
    #find bin centres for plotting
    bin_centres = np.zeros(bins)
    for i in range(bins):
        bin_centres[i] = (bin_edges[i] + bin_edges[i+1]) / 2
    #plot histogram of noise data
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centres, counts, bin_width, label='Histogram', edgecolor='k')   # or plt.step(...)

    #plot gaussian over histogram
    x = np.linspace(min, max, 100)
    #gaussian formula provides pdf, not frequency, so the values must be scaled
    density = (1/(np.sqrt(var*2*np.pi))) * np.exp(-0.5 * ((x - mean)**2) / var)
    freq = density * len(data) * bin_width

    plt.plot(x, freq, label='Gaussian', color='red')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def get_data_sec_III(accel_filepath, gyro_filepath, gps_filepath):
    """"
    loads and corrects accel, gyro, and gps data
    """
    accel_measured = read_csv(accel_filepath, maxrows=15000)
    accel_measured[:,3] -= 9.805
    gyro_measured = read_csv(gyro_filepath, maxrows=15000)
    gps = read_gps_csv(gps_filepath, maxrows=15000)

    X_acc = np.column_stack((np.ones(accel_measured.shape[0]), accel_measured[:,0]))
    X_gyr = np.column_stack((np.ones(gyro_measured.shape[0]), gyro_measured[:,0]))

    accel = accel_measured.copy()
    gyro = gyro_measured.copy()
    
    accel[:,1:] -= np.column_stack((X_acc @ beta_x, X_acc @ beta_y, X_acc @ beta_z))
    gyro[:,1:] -= np.column_stack((X_gyr @ beta_i, X_gyr @ beta_j, X_gyr @ beta_k))

    return accel, gyro, gps

def convert_gps(lat, long):
    """
    using lat0, long0 as the origin:

    x = (long-long0) cos(lat0) * R
    y = (lat - lat0) * R
    """
    R = 6.378E6
    lat = lat*np.pi/180
    long = long*np.pi/180
    lat0 = lat[0]
    long0 = long[0]

    x = (long - long0) * np.cos(lat0)*R
    y = (lat - lat0)*R
    return x, y


# def return_biases(data):

#     #dimensions
#     m = data.shape[0]
#     n = 2

#     #using linear least squares definition: beta = (x.T * x)^-1 * x.T * y
#     X = np.column_stack((np.ones(m), data[:,0]))

#     #find betas for each direction
#     beta_x = np.linalg.inv((X.T @ X)) @ X.T @ data[:,1]
#     beta_y = np.linalg.inv((X.T @ X)) @ X.T @ data[:,2]
#     beta_z = np.linalg.inv((X.T @ X)) @ X.T @ data[:,3]

#     bias_x = X @ beta_x
#     bias_y = X @ beta_y
#     bias_z = X @ beta_z
#     biases = np.column_stack((data[:,0],bias_x, bias_y, bias_z))
#     return biases

# def return_unbiased(data):
#     #dimensions
#     m = data.shape[0]
#     n = 2

#     #using linear least squares definition: beta = (x.T * x)^-1 * x.T * y
#     X = np.column_stack((np.ones(m), data[:,0]))

#     #find betas for each direction
#     beta_x = np.linalg.inv((X.T @ X)) @ X.T @ data[:,1]
#     beta_y = np.linalg.inv((X.T @ X)) @ X.T @ data[:,2]
#     beta_z = np.linalg.inv((X.T @ X)) @ X.T @ data[:,3]

#     bias_x = X @ beta_x
#     bias_y = X @ beta_y
#     bias_z = X @ beta_z
#     biases = np.column_stack((data[:,0],bias_x, bias_y, bias_z))
#     return np.column_stack((data[:,0],data[:,1]-bias_x, data[:,2]-bias_y, data[:,3]-bias_z))
