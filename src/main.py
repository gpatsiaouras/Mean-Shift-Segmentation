import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Segmentation:
    def __init__(self, data, radius):
        self.data = data
        self.radius = radius
        self.conversion_threshold = self.radius/2
        self.threshold = 0.01
        self.peaks = []
        self.labels = np.zeros(data.shape[1])

    def find_peak(self, previous_data_point):
        neighbors = self.get_data_in_radius_from_point(previous_data_point)
        mean_data_point = np.mean(neighbors, axis=1)
        while not self.converged(mean_data_point, previous_data_point):
            previous_data_point = mean_data_point
            neighbors = self.get_data_in_radius_from_point(mean_data_point)
            mean_data_point = np.mean(neighbors, axis=1)

        return mean_data_point

    def mean_shift(self):
        for i in range(self.data.shape[1]):
            peak = self.find_peak(self.data[:, i])
            self.labels[i] = self.get_label_for_point(peak)
            sys.stdout.write("\rPoint: {0}, Peaks: {1}".format(i, len(self.peaks)))
            sys.stdout.flush()

    def get_data_in_radius_from_point(self, cluster_point):
        subdata = []
        for i in range(self.data.shape[1]):
            if self.get_euclidean_distance(cluster_point, self.data[:, i]) < self.radius:
                subdata.append(self.data[:, i])

        return np.array(subdata).T

    def get_euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(
            (x2 - x1) ** 2
        ))

    def converged(self, mean_data_point, previous_data_point):
        return self.get_euclidean_distance(mean_data_point, previous_data_point) < self.threshold

    def get_label_for_point(self, new_peak):
        for peak_idx in range(len(self.peaks)):
            if (np.abs(new_peak - self.peaks[peak_idx]) < self.conversion_threshold).all():
                return peak_idx

        self.peaks.append(new_peak)

        return len(self.peaks) - 1


if __name__ == "__main__":
    # Read data
    data = np.array(scipy.io.loadmat('pts.mat')['data'])
    # np.random.shuffle(data.T)
    # data = data[:, :100]
    image_seg = Segmentation(data, 2)
    image_seg.mean_shift()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2])
    peaks = np.array(image_seg.peaks).T
    ax.scatter(peaks[0], peaks[1], peaks[2])
    plt.show()

    # print(image_seg.labels)
