"""
Create a reference landmark file from a SNIRF file.

Purpose
-------
Extract 3D landmark coordinates and labels from a reference SNIRF
recording and save them to a .npz file for later use.

Inputs
------
Reference SNIRF file containing known landmarks.

Outputs
-------
NPZ file containing 3D landmark positions and landmark labels.

Author
------
Emma Bailey, 30.07.2026
"""

import os
import numpy as np
from snirf import Snirf


def create_landmark_file(reference_snirf_file):
    """
    Extract landmarks from a reference SNIRF file and save them to a cache file.

    Parameters
    ----------
    reference_snirf_file : str
        Path to the reference SNIRF file.

    Returns
    -------
    None
        The extracted landmark data are saved to a .npz file.
    """

    with Snirf(reference_snirf_file, 'r') as reference_snirf:
        reference_probe = reference_snirf.nirs[0].probe
        landmark_pos_3d = np.array(reference_probe.landmarkPos3D)
        landmark_labels = np.array(reference_probe.landmarkLabels)

    np.savez(landmark_cache_file, landmark_pos_3d=landmark_pos_3d, landmark_labels=landmark_labels)
    print(f"Cached reference landmarks to {landmark_cache_file}")


def load_landmarks(landmark_cache_file):
    """
    Load landmark positions and labels from a cached landmark file.

    Parameters
    ----------
    landmark_cache_file : str
        Path to the cached .npz landmark file.

    Returns
    -------
    landmark_pos_3d : numpy.ndarray
        Three-dimensional landmark coordinates.
    landmark_labels : numpy.ndarray
        Labels corresponding to the landmarks.
    """

    if os.path.exists(landmark_cache_file):
        with np.load(landmark_cache_file, allow_pickle=False) as cache:
            return cache["landmark_pos_3d"], cache["landmark_labels"]

    else:
        raise RuntimeError(f"Landmark cache file {landmark_cache_file} does not exist. ")

    return landmark_pos_3d, landmark_labels


if __name__ == '__main__':
    # reference_snirf_file and landmark_cache_file are hard-coded here; they must be changed as needed.
    reference_snirf_file = "C:\\Users\\Emma\\Documents\\NIRx\\Data\\aurora_2025_2\\2025-10-29_002.snirf"
    landmark_cache_file = "C:\\Users\\Emma\\OneDrive - NIRx Medizintechnik GmbH\\Documents\\Montage_Design\\Montages\\snirf_from_probeInfo\\icbm152_landmarks.npz"
    
    # Example call to create the landmark cache file from a reference SNIRF file
    create_landmark_file(reference_snirf_file)

    # Example call on how to load from a cache file
    landmark_pos_3d, landmark_labels = load_landmarks(landmark_cache_file)