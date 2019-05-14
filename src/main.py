import time
import cv2
import random
import numpy as np
import scipy.io
from skimage import io, color
import matplotlib.pyplot as plt
import progressbar
from mpl_toolkits.mplot3d import Axes3D


def plot_data_points_and_peaks(data, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2])
    peaks = np.array(model.peaks).reshape(len(model.peaks), 3).T
    ax.scatter(peaks[0], peaks[1], peaks[2])
    plt.show()


class MeanShiftSegmentation:
    def __init__(self, data, radius, c):
        """
        Assigns data and radius
        Initiates a threshold and a conversion threshold, an empty list to save peaks,
        an vector of labels with default value -1
        :param data:
        :param radius:
        """
        self.data = data
        self.radius = radius
        self.c = c
        self.conversion_threshold = self.radius / 2
        self.threshold = 0.01
        self.peaks = []
        # Initiate labels with -1 values indicating that there is no label assigned yet
        self.labels = np.ones(data.shape[1]) * -1

    def find_peak(self, previous_data_point):
        """
        Applies iterative process that based on a sphere of data generated by a radius
        it finds the mean point and then shifts the sphere to the new mean point until the
        sphere stops moving.
        :param previous_data_point:
        :return mean_data_point: Peak found
        """
        neighbors, points_in_radius = self.get_data_in_radius_from_point(previous_data_point)
        mean_data_point = np.mean(neighbors, axis=1).reshape(self.data.shape[0], 1)
        while not self.converged(mean_data_point, previous_data_point):
            previous_data_point = mean_data_point
            neighbors, points_in_radius = self.get_data_in_radius_from_point(mean_data_point)
            mean_data_point = np.mean(neighbors, axis=1).reshape(self.data.shape[0], 1)

        return mean_data_point

    def find_peak_opt(self, previous_data_point):
        """
        Finds peak point as find_peak but also returns the points that should be associated with the label
        defined by this peak.
        :param previous_data_point:
        :return mean_data_point, points_in_radius: Peak found and points at range
        """
        # Creating an array of the same feature size as data with False default value indicating that
        # no feature should be associated with label.
        points_to_be_associated = np.full((self.data.shape[1],), False, dtype=bool)

        points_values, points_keys = self.get_data_in_radius_from_point(previous_data_point)
        mean_data_point = np.mean(points_values, axis=1).reshape(self.data.shape[0], 1)

        while not self.converged(mean_data_point, previous_data_point):
            previous_data_point = mean_data_point

            # Get the new points inside the sphere
            points_values, points_keys = self.get_data_in_radius_from_point(mean_data_point)

            # Second optimization. Take the points in range radius/c
            points_keys_2 = self.get_point_keys_in_radius(mean_data_point, self.radius / self.c)

            # Second optimization Save the path of the algorithm
            points_to_be_associated = points_to_be_associated | points_keys_2

            # Recalculate mean point
            mean_data_point = np.mean(points_values, axis=1).reshape(self.data.shape[0], 1)

        # In the existing point keys also add the data in radius distance from the last mean point.
        # as instructed by the first optimization
        points_to_be_associated = points_to_be_associated | points_keys
        return mean_data_point, points_to_be_associated

    def mean_shift(self):
        """
        Mean shift implementation, iterates over data points and for each one it finds
        the peak and assigns the approriate label.
        """
        start_time = time.time()
        print("Running mean shift algorithm")
        for i in progressbar.progressbar(range(self.data.shape[1]), redirect_stdout=True):
            peak = self.find_peak(self.data[:, i].reshape(self.data.shape[0], 1))
            self.labels[i] = self.get_label_for_point(peak)

        print("\rPeaks found: {0}, Exec Time Mean Shift: {1:.2f} seconds"
              .format(len(self.peaks), time.time() - start_time) * 1000)

    def mean_shift_opt(self):
        """
        Optimized mean shift algorithm applying optimization of basin of attraction and
        points along the search path association with the converged peak.
        """
        start_time = time.time()
        print("Running mean shift algorithm optimized")
        for i in progressbar.progressbar(range(self.data.shape[1]), redirect_stdout=True):
            if self.labels[i] == -1:
                peak, points_in_radius = self.find_peak_opt(self.data[:, i].reshape(self.data.shape[0], 1))
                self.labels[i] = self.get_label_for_point(peak)
                self.labels[points_in_radius] = self.labels[i]

        print("\rPeaks found: {0}, Exec Time Optimized Mean Shift: {1:.2f} seconds"
              .format(len(self.peaks), time.time() - start_time) * 1000)

    def get_data_in_radius_from_point(self, cluster_point):
        """
        Applies a sphere with specified radius and filters and returns the data points
        belonging to this sphere. It also returns the conditional array holding the indices
        of each point from the original array and True or False, indicating if they belong
        to the current sphere or not.
        :param cluster_point: Center of sphere
        :return points_in_range, indices_of_points: Points in range their indices
        """
        points_in_radius = self.get_point_keys_in_radius(cluster_point, self.radius)
        extracted = np.extract(np.tile(points_in_radius, (3, 1)), self.data)
        return extracted.reshape(3, extracted.shape[0] // 3), points_in_radius

    def get_point_keys_in_radius(self, cluster_point, radius):
        """
        Returns indices of data that are inside radius of the cluster points by
        calculating their euclidean distance.
        :param cluster_point: Center of Sphere
        :param radius: Radius to be appllied
        :return: array with true false values whether the element belongs to sphere or not
        """
        return np.linalg.norm(cluster_point - self.data, axis=0) < radius

    def converged(self, mean_data_point, previous_data_point):
        """
        Checks if the mean data point is close enough to the previous mean data point
        and returns true if it is less than the threshold.
        :param mean_data_point: Current center of sphere
        :param previous_data_point: Previous center of sphere
        :return True or False: True if it converged, False if not
        """
        return np.linalg.norm(mean_data_point - previous_data_point) < self.threshold

    def get_label_for_point(self, new_peak):
        """
        Searches if the new_peak found already exists in the peaks list by
        using a threshold between the values. If the peak exists it returns the index
        of the peak, otherwise adds the peak to the list and returns last index.
        :param new_peak: New peak found for this point
        :return index of peak: Index of the peak in the peaks list
        """
        for peak_idx in range(len(self.peaks)):
            if (np.abs(new_peak - self.peaks[peak_idx]) < self.conversion_threshold).all():
                return peak_idx

        self.peaks.append(new_peak)

        return len(self.peaks) - 1

    def im_segment(self, image, radius):
        pass


if __name__ == "__main__":
    # Read debug data
    data = np.array(scipy.io.loadmat('../resources/pts.mat')['data'])

    # Read image
    image = cv2.imread("../resources/55075.jpg")
    cv2.imshow('Original image', image)

    image = color.rgb2lab(image)
    image_array = np.zeros((3, image.shape[0] * image.shape[1]))
    l, a, b = cv2.split(image)
    image_array[0, :] = l.flatten()
    image_array[1, :] = a.flatten()
    image_array[2, :] = b.flatten()

    # Run Without Optimization
    # image_seg = MeanShiftSegmentation(data, radius=2, c=4)
    # image_seg.mean_shift()

    # Run With Optimization
    image_seg_opt = MeanShiftSegmentation(image_array, radius=30, c=10)
    image_seg_opt.mean_shift_opt()

    # Plot data points
    # plot_data_points_and_peaks(image_array, image_seg_opt)

    overlay = np.zeros((3, image_seg_opt.labels.shape[0]))

    # for segment in np.unique(image_seg_opt.labels):
    for segment in np.unique(image_seg_opt.labels):
        overlay[:, image_seg_opt.labels == segment] = np.average(image_array[:, image_seg_opt.labels == segment])

    overlay = cv2.merge((overlay[0, :], overlay[1, :], overlay[2, :]))
    overlay = overlay.reshape(image.shape[0], image.shape[1], 3)
    overlay = color.lab2rgb(overlay)

    cv2.imwrite('../out/overlay.jpg', overlay * 255)
    cv2.imshow('Segmented image', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
