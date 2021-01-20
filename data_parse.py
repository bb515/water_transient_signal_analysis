"""Experimental data read from text file."""
import numpy as np
import h5py
import warnings
import re
import matplotlib.pyplot as plt
import pathlib
from scipy.fft import fft, ifft

write_path = pathlib.Path()


def read_txt(file_name):
    """Read .txt file and turn the columns into np.ndarray."""
    time_stamps = []
    pressures = []
    with open(file_name, 'r') as f:
        # read until we hit time_stamp
        while True:
            line = f.readline()
            print(line)
            if ("value" in line):
                print('1')
                break

        while True:
            line = f.readline()
            # remove the trailing newline char
            line = re.sub("\n", "", line)
            # remove surrounding whitespace
            line = line.strip()
            # check for EOF
            if line == '':
                break
            cols = re.split("\s+", line)
            print(cols)
            # date = float(cols[0])
            time = cols[1]
            time = time.split(':')
            print(time)
            minutes = int(time[1])
            seconds = float(time[2])
            time_stamp = float(minutes * 60 + seconds)
            pressure = float(cols[2])
            time_stamps.append(time_stamp)
            pressures.append(pressure)

    time_stamps = np.array(time_stamps)
    pressures = np.array(pressures)

    return time_stamps, pressures


def write_array(write_path, dataset, array):
    """
    Write a :class: numpy.ndarray to a HDF5 file.

    :arg write_path: The path to which the HDF5 file is written.
    :type write_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :array: The array to be written to file.
    :type array: :class: numpy.ndarray

    :return: None
    :rtype: None type
    """
    with h5py.File(write_path, 'a') as hf:
        hf.create_dataset(dataset,  data=array)


def read_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.

    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str

    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                array = hf[dataset][:]
                return array
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
    except IOError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        return None


def fft_plot(y, t):
    """Compute the fast fourier transform."""
    # Compute the mean
    mean_y = np.mean(y)
    N = np.shape(y)[0]
    y = y - mean_y * np.ones(N)
    T = t[1] - t[0]
    Total_time = t[-1] - t[0]
    print(T, 'T')
    x = np.linspace(0.0, Total_time, N)
    print(x[:20])
    plt.plot(x, y)
    plt.show()
    yf = fft(y)
    yf = 2.0/ N * np.abs(yf[0:N//2])
    xf = np.linspace(0.0, Total_time/(2.0*T), N//2)
    plt.plot(xf, yf)
    sorted_xf = xf[yf.argsort()]
    sorted_yf = yf[yf.argsort()]
    print(sorted_yf[-10:], 'sorted_yf')
    print(sorted_xf[-10:], 'sorted_xf')
    plt.xlim(0.0, 100)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('fourier coefficient magnitude')
    plt.savefig('fourier1.png')
    plt.show()
    return print('0')

def reconstruct(y, t):
    """Reconstruct a signal from fourier components"""
    # Generate basis functions
    # dt = 0:1 / 60: 3;
    # df = [3:3: 12];

    # basis1 = np.exp(1j * 2 * pi * 3 * dt);
    # basis2 = np.exp(1j * 2 * pi * 6 * dt);
    # basis3 = np.exp(1j * 2 * pi * 9 * dt);
    # basis4 = np.exp(1j * 2 * pi * 12 * dt);
    #
    # % Reconstruct
    # var
    # var_recon = basis1 * f_useful(1) + ...
    # basis2 * f_useful(2) + ...
    # basis3 * f_useful(3) + ...
    # basis4 * f_useful(4);
    # var_recon = real(var_recon);

    # Plot both curves
    # plt.plot(var)
    # plt.plot(var_recon)



    # Compute the mean
    mean_y = np.mean(y)
    N = np.shape(y)[0]
    y_true = y - mean_y * np.ones(N)
    T = t[1] - t[0]
    Total_time = t[-1] - t[0]
    print(T, 'T')
    x = np.linspace(0.0, Total_time, N)
    print(x)
    f1 = 10.00978474
    c1 = 2.55180071
    w1 = np.pi * 2 * f1
    y_plot = c1 * np.sin(f1 * x)
    plt.scatter(x, y_plot)
    plt.show()
    # plt.plot(x, y)
    # plt.plot(x, y_true)
    # plt.grid()
    # plt.show()
    return None


# Calculate the moments


#time_stamps, pressure = read_txt(
#    write_path/"trimble.txt")
#write_array(write_path/"trimble.h5", "t", time_stamps)
#write_array(write_path/"trimble.h5", "p", pressure)
#np.savez(write_path/"PT_sample", t=time_stamps, p=pressure)



# Load the data.
data = np.load(write_path/"PT_sample.npz")
p = data['p']
t = data['t']


x = np.linspace(0, 10)
y = np.sin(x)
plt.plot(x,y)
plt.show()

plt.plot(t, p)
plt.show()




# Plot the fourier transform
#fft_plot(p, t)
# reconstruct(p, t)
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# plt.plot(x,y)
# plt.show()
# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# import matplotlib.pyplot as plt
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(time_stamps, pressure)
#
# textstr = '\n'.join((
#     'impulse_data_id = 486518',
#     'measurement_id = 102032',
#     'start_time = 2020-11-16 6:11:45',
#     'end_time = 2020-11-16 6:16:01',
#     'duration = 256',
#     'min_pressure = 52.38462',
#     'avg_pressure = 85.28192982',
#     'max_pressure = 92.38462',
#     'severity = 0.9380650763',
#     'rank =  4',
#     'time_of_min = 6:12:44',
#     'time_of_max = 6:13:27'))
#
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#
# # place a text box in upper left in axes coords
# ax.text(0.05, 0.55, textstr, transform=ax.transAxes, fontsize=7,
#         verticalalignment='top', bbox=props)

# plt.xlabel('seconds (s)')
# plt.ylabel('pressure ')
# plt.savefig('trimble_data.png')
# plt.show()
