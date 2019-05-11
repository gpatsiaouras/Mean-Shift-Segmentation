import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Segmentation:
    def __init__(self, data, radius):
        self.data = data
        self.radius = radius
        self.previous_data_point = None
        self.conversion_threshold = self.radius/2
        self.threshold = 0.01
        self.peaks = []

    def find_peak(self, subdata):
        mean_data_point = np.mean(subdata, axis=1)
        if self.previous_data_point is None or not self.is_converged(mean_data_point):
            self.previous_data_point = mean_data_point
            new_subdata = self.get_sphere(mean_data_point)
            return self.find_peak(new_subdata)
        else:
            return mean_data_point

    def mean_shift(self):
        for i in range(self.data.shape[1]):
            subdata = self.get_sphere(self.data[:, i])
            peak = self.find_peak(subdata)
            self.add_peak_to_list(peak)
            sys.stdout.write("\rPoint: {0}, Peaks: {1}".format(i, len(self.peaks)))
            sys.stdout.flush()

    def get_sphere(self, cluster_point):
        subdata = []
        for i in range(self.data.shape[1]):
            if self.get_euclidean_distance(cluster_point, self.data[:, i]) < self.radius:
                subdata.append(self.data[:, i])

        return np.array(subdata).T

    def get_euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(
            (x2 - x1) ** 2
        ))

    def is_converged(self, mean_data_point):
        if self.previous_data_point is not None:
            return (np.abs(mean_data_point - self.previous_data_point) < self.conversion_threshold).all()
        return False

    def add_peak_to_list(self, new_peak):
        found_similar_peak = False
        for peak in self.peaks:
            diff = np.abs(new_peak - peak)
            if (diff < self.threshold).all():
                found_similar_peak = True
                break

        if not found_similar_peak:
            self.peaks.append(new_peak)


if __name__ == "__main__":
    # Read data
    data = np.array(scipy.io.loadmat('pts.mat')['data'])

    image_seg = Segmentation(data, 2)
    image_seg.mean_shift()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2])
    peaks = np.array(image_seg.peaks).T
    ax.scatter(peaks[0], peaks[1], peaks[2])
    plt.show()
