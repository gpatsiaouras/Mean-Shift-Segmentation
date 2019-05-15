# Mean Shift Segmentation

## Description
The mean-shift algorithm clusters an n-dimensional data set by associating each point to a peak
of the data set's probability density. For each point, mean-shift computes its associated peak by
first defining a spherical window at the data point of radius r and computing the mean of the points
that lie within the window. The algorithm then shifts the window to the mean and repeats until
convergence, i.e. the shift is under some threshold t (for example, t = 0.01). With each iteration the
window will shift to a more densely populated portion of the data set until a peak is reached, where
the data is equally distributed in the window.

To speed up the original algorithm we are going to implement two speedups:
1. The first speedup will be to associate each data point that is at a distance
≤ r from the peak with the cluster defined by that peak. This speedup is known as basin of
attraction and is based on the intuition that points that are within one window size distance from
the peak will with high probability converge to that peak.

2. The second speedup is based on a similar principle, where points that are within a distance of
r/c of the search path are associated with the converged peak, where c is some constant value
(check Figure 1(b)). We will choose c = 4 for this problem but you are also asked to check other
values. Incorporate the above speedups into your mean-shift implementation by modifying your
implementation from the previous part.

The algorithm can run in 3 dimensions (RGB) for each point(pixel) of the image but also in 5d
(RGBxy) where x,y are dimensions of the point in the image. With this approach the distance of the pixels
from each other is also taken into consideration to avoid, associating pixels with peaks far away when color
is similar.

## Prepare Environment
The algorithm is using a few libraries to make the approriate computations, 
install them using this command.

    $ pip3 install -r requirements.txt
    
## Run
To run cd inside src directory and execute using python3

    $ cd src/
    $ python3 main.py