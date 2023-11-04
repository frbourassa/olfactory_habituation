import numpy as np


def moving_average(points, kernelsize, boundary="free"):
    """ Moving average filtering on the array of experimental points,
    averages over a block of size kernelsize.
    kernelsize should be an odd number; otherwise,
    the odd number just lower is used.
    The ith smoothed value, S_i, is:
        $$ S_i = \frac{1}{kernelsize} \sum_{j = i-kernelsize//2}^{i + kernelsize//2} x_j $$
    Values at the boundary are smoothed with smaller and smaller kernels
    (up to size 1 for boundary values)

    Args:
        points (1darray): the experimental data points
        kernelsize (int): odd integer giving the total number of points summed
        boundary (str): how to deal with points within kernelsize//2 of edges
            "shrink": the window for a point within distance d < w
                is shrunk symmetrically to a kernel of size d
            "free": the window is asymmetric, full on the inside and clipped
                on the side near the edge.
            "noflux": these points are set to the value of the closest point
                with full window (i.e. distance kernelsize//2 of the edge)

    Returns:
        smoothed (ndarray): the smoothed data points.
    """
    smoothed = np.zeros(points.shape)
    if kernelsize % 2 == 0:  # if an even number was given
        kernelsize -= 1
    w = kernelsize // 2  # width
    end = smoothed.shape[0]  # index of the last element

    if boundary not in ["shrink", "free", "noflux"]:
        raise ValueError("Unknown boundary {}".format(boundary))

    # Smooth the middle points using slicing.
    smoothed[w:end - w] = points[w:end - w]
    for j in range(w):  # Add points around the middle one
        smoothed[w:-w] += points[w - j - 1:end - w - j - 1] + points[w + j + 1:end - w + j + 1]

        # Use the loop to treat the two points at a distance j from boundaries
        if j < w and boundary == "shrink":
            smoothed[j] = np.sum(points[0:2*j + 1], axis=0) / (2*j + 1)
            smoothed[-j - 1] = np.sum(points[-2*j - 1:], axis=0)/(2*j + 1)
        elif j < w and boundary == "free":
            smoothed[j] = np.sum(points[0:j + w + 1], axis=0) / (j + w + 1)
            smoothed[-j - 1] = np.sum(points[-j - w - 1:], axis=0) / (j + w + 1)

    # Normalize the middle points
    smoothed[w:end - w] = smoothed[w:end - w] / kernelsize

    # If noflux boundary, set edge points
    if boundary == "noflux":
        smoothed[:w] = smoothed[w]
        smoothed[-w:] = smoothed[-w - 1]

    return smoothed


def moving_var(points, kernelsize, ddof=1, boundary="free"):
    """ Similar to moving_average, but computing the variance
    (instead of the average) in a sliding window.

    Args:
        points (np.ndarray): the data points
        kernelsize (int): odd integer giving the total number of points
        boundary (str): how to deal with points within kernelsize//2 of edges
            "shrink": the window for a point within distance d < w
                is shrunk symmetrically to a kernel of size d
            "free": the window is asymmetric, full on the inside and clipped
                on the side near the edge.
            "noflux": these points are set to the value of the closest point
                with full window (i.e. distance kernelsize//2 of the edge)

    Returns:
        var_points (np.ndarray): standard deviation at every point
    """
    var_points = np.zeros(points.shape)
    # To compute std, we need to compute the average too
    avg_points = np.zeros(points.shape)
    if kernelsize < 3: raise ValueError("Need larger kernel for variance")
    if kernelsize % 2 == 0:  # if an even number was given
        kernelsize -= 1
    w = kernelsize // 2  # width
    end = avg_points.shape[0]  # index of the last element

    if boundary not in ["shrink", "free", "noflux"]:
        raise ValueError("Unknown boundary {}".format(boundary))

    # Smooth the middle points using slicing.
    # First store second moment in var_points
    var_points[w:end - w] = points[w:end - w]**2
    avg_points[w:end - w] = points[w: end - w]
    for j in range(w):  # Add points around the middle one
        avg_points[w:-w] += points[w - j - 1:end - w - j - 1]
        avg_points[w:-w] += points[w + j + 1:end - w + j + 1]
        var_points[w:-w] += points[w - j - 1:end - w - j - 1]**2
        var_points[w:-w] += points[w + j + 1:end - w + j + 1]**2

        # Use the loop to treat the two points at a distance j from boundaries
        if j < w and j > 0 and boundary == "shrink":
            avg_points[j] = np.sum(points[0:2*j + 1], axis=0) / (2*j + 1)
            var_points[j] = (np.sum(points[0:2*j + 1]**2, axis=0)
                    - avg_points[j]**2 * (2*j + 1)) / (2*j + 1 - ddof)
            avg_points[-j - 1] = np.sum(points[-2*j - 1:], axis=0) / (2*j + 1)
            var_points[-j - 1] = (np.sum(points[-2*j - 1:]**2, axis=0)
                    - avg_points[-j - 1]**2 * (2*j + 1)) / (2*j + 1 - ddof)
        elif j < w and boundary == "free":
            avg_points[j] = np.sum(points[0:j + w + 1], axis=0) / (j + w + 1)
            var_points[j] = (np.sum(points[0:j + w + 1]**2, axis=0)
                    - avg_points[j]**2 * (j + w + 1)) / (j + w + 1 - ddof)
            avg_points[-j - 1] = np.sum(points[-j - w - 1:], axis=0) / (j + w + 1)
            var_points[-j - 1] = (np.sum(points[-j - w - 1:]**2, axis=0)
                    - avg_points[-j - 1]**2 * (j + w + 1)) / (j + w + 1 - ddof)

    # Normalize the middle points by kernelsize - ddof
    avg_points[w:end - w] /= kernelsize
    var_points[w:end - w] /= (kernelsize - ddof)

    # Set the edge points to the nearest full point if boundary is no flux
    if boundary == "noflux":
        var_points[:w] = var_points[w]
        var_points[-w:] = var_points[-w]

    # Then subtract the average squared, taking ddof into account once
    var_points[w:end - w] -= (avg_points[w:end - w]**2
                                * kernelsize / (kernelsize - ddof))

    return var_points


# Small utility functions
def find_nearest(my_array, target, condition):
    """ Nice function by Akavall on StackOverflow:
    https://stackoverflow.com/questions/17118350/how-to-find-nearest-value-that-is-greater-in-numpy-array
    Page consulted Feb 10, 2019.
    """
    diff = my_array - target
    if condition == "above":
        # We need to mask the negative differences and zero
        # since we are looking for values above
        mask = (diff <= 0)
    elif condition == "below":
        # We need to mask positive differences and zero
        mask = (diff >= 0)
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()
