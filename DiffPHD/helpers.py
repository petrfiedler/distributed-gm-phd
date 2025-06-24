import numpy as np
from matplotlib.patches import Ellipse


def bhattacharyya_distance(
    mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray
) -> float:
    """
    Calculate the Bhattacharyya distance between two Gaussian distributions.

    :param mean1: The mean of the first Gaussian distribution.
    :param cov1: The covariance of the first Gaussian distribution.
    :param mean2: The mean of the second Gaussian distribution.
    :param cov2: The covariance of the second Gaussian distribution.
    :returns: The Bhattacharyya distance between the two distributions.
    """
    assert mean1.shape == mean2.shape
    assert cov1.shape == cov2.shape
    assert cov1.shape[0] == cov1.shape[1]
    assert cov2.shape[0] == cov2.shape[1]
    assert cov1.shape[0] == mean1.shape[0]
    assert cov2.shape[0] == mean2.shape[0]

    mean_diff = mean1 - mean2
    cov_avg = (cov1 + cov2) / 2 + np.eye(mean1.shape[0]) * np.finfo(float).eps
    cov_avg_inv = np.linalg.inv(cov_avg)
    det_cov_avg = np.linalg.det(cov_avg)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)

    return 0.125 * mean_diff.T @ cov_avg_inv @ mean_diff + 0.5 * np.log(
        det_cov_avg / np.sqrt(det_cov1 * det_cov2)
    )


def plot_cov_ellipse(mean, cov, n_std=3, **kwargs):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    width = n_std * np.sqrt(eigenvalues[0])
    height = n_std * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    return Ellipse(
        xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs
    )
