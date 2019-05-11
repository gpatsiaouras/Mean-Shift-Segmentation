import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Segmentation:
    def __init__(self, data, radius):
        self.data = data
        self.radius = radius

    def find_peak(self, subdata, idx, radius):
        pass

    def mean_shift(self):
        for i in range(len(self.data.shape[1])):
            data = self.get_sphere(data[i])
            peak = self.find_peak(data, (idy, idx), self.radius)

    def get_sphere(self, cluster_point):
        

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(
            (x2 - x1) ** 2
        ))


if __name__ == "__main__":
    # Read data
    data = np.array(scipy.io.loadmat('pts.mat')['data'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2])
    plt.show()
    # image_seg = Segmentation(data)
