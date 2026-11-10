"""
Create SNIRF files from NIRSite probeInfo.json files.

Purpose
-------
Generate SNIRF files containing the 2D and 3D source and detector
positions required for sensitivity-profile generation.

Inputs
------
- NIRSite 2026.2 probeInfo.json files.
- Cached ICBM152 landmark positions and labels.

Outputs
-------
A SNIRF file including probe information for each specified montage. No actual data is included.

Author
------
Emma Bailey, 31.07.2026
"""


import json
import os
from pathlib import Path

import numpy as np
from snirf import Snirf
from create_landmark_file import load_landmarks

# NIRx Aurora systems record at these two wavelengths; probeInfo.json does not itself carry this information
WAVELENGTHS_NM = [760.0, 850.0]


def create_probeinfo_snirf(probe_file, out_file, landmark_pos_3d, landmark_labels):
    """Create a SNIRF file from a NIRSite probeInfo.json file.

    Parameters
    ----------
    probe_file : str
        Path to the input probeInfo.json file.
    out_file : str
        Path for the generated SNIRF file.
    landmark_pos_3d : numpy.ndarray
        Three-dimensional reference landmark positions.
    landmark_labels : numpy.ndarray
        Labels corresponding to the reference landmarks.

    Returns
    -------
    None
        The generated SNIRF file is written to disk.
    """

    # Load probeInfo.json file
    with open(probe_file, "r") as f:
        probe_info = json.load(f)

    # Extract probe information
    probes = probe_info["probeInfo"]["probes"]
    length_unit = probe_info["probeInfo"]["length_unit"]
    coords_s2 = np.array(probes["coords_s2"])
    coords_s3 = np.array(probes["coords_s3"])
    coords_d2 = np.array(probes["coords_d2"])
    coords_d3 = np.array(probes["coords_d3"])
    index_c = probes["index_c"]  # [sourceIndex, detectorIndex] per channel, 1-based

    # Remove output file if it exists
    if os.path.exists(out_file):
        os.remove(out_file)

    # Instantiate a new SNIRF file and populate it with the probe information
    snirf = Snirf(out_file, 'w')
    snirf.formatVersion = "1.1"

    snirf.nirs.appendGroup()
    nirs = snirf.nirs[0]

    # Minimal required metadata
    nirs.metaDataTags.SubjectID = "unknown"
    nirs.metaDataTags.MeasurementDate = "unknown"
    nirs.metaDataTags.MeasurementTime = "unknown"
    nirs.metaDataTags.LengthUnit = length_unit
    nirs.metaDataTags.TimeUnit = "s"
    nirs.metaDataTags.FrequencyUnit = "Hz"

    # Probe geometry and information
    nirs.probe.sourcePos2D = coords_s2
    nirs.probe.sourcePos3D = coords_s3
    nirs.probe.detectorPos2D = coords_d2
    nirs.probe.detectorPos3D = coords_d3
    nirs.probe.wavelengths = np.array(WAVELENGTHS_NM)

    # Set landmark information from the cached landmark file, 
    # which was created from a reference SNIRF file with known landmarks.
    nirs.probe.landmarkPos3D = landmark_pos_3d
    nirs.probe.landmarkLabels = landmark_labels
    
    # probeInfo.json reports "MNI" against the ICBM152 head model, which isn't one
    # of the exact BIDS-standard coordinateSystem tags, so record it as "Other" with
    # a description instead of a value the SNIRF spec doesn't recognize.
    nirs.probe.coordinateSystem = "Other"
    nirs.probe.coordinateSystemDescription = (
        f"{probe_info['probeInfo']['coordinate_system']} space, "
        f"{probe_info['probeInfo']['headmodel']} head model"
    )

    # No real data required, but Cedalion requires SNIRF to have a valid data block with a
    # measurementList entry per channel/wavelength, so populate it with
    # placeholder zeros using the real source-detector pairing from index_c.
    # At least 2 timepoints are needed to read in Cedalion
    nirs.data.appendGroup()
    data = nirs.data[0]
    n_channels = len(index_c) * len(WAVELENGTHS_NM)
    data.dataTimeSeries = np.zeros((2, n_channels))
    data.time = np.array([0.0, 1.0])

    for source_index, detector_index in index_c:
        for wavelength_index in range(1, len(WAVELENGTHS_NM) + 1):
            data.measurementList.appendGroup()
            ml = data.measurementList[-1]
            ml.sourceIndex = source_index
            ml.detectorIndex = detector_index
            ml.wavelengthIndex = wavelength_index
            ml.dataType = 1
            ml.dataTypeLabel = "raw-DC"
            ml.dataTypeIndex = 1
            ml.dataUnit = "mV"

    # Validate created snirf file
    result = snirf.validate()
    print(result.display())

    # Save and close
    snirf.save()
    snirf.close()
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    # Montage names and cache file location are hardcoded here; they must be changed as needed.
    montage_names = [
        "Standard_AudioVisualSDD_16x16", 
        "Standard_fNIRSEEG_32x32", 
        "Standard_FrontalOccipitalSDD_16x16", 
        "Standard_FrontalTemporalSDD_16x16",
        "Standard_FrontalSDD_16x16", 
        "Standard_MotorSDD_8x8", 
        "Standard_MotorSDD_16x16", 
        "Standard_PrefrontalMotorSDD_16x16", 
        "Standard_WholeHeadSDD_32x32", 
        "Standard_WholeHeadSDD_48x48"
    ]
    landmark_cache_file = "C:\\Users\\Emma\\OneDrive - NIRx Medizintechnik GmbH\\Documents\\Montage_Design\\Montages\\snirf_from_probeInfo\\icbm152_landmarks.npz"

    # Same landmark grid (ICBM152 template) for every montage below, so this only needs 
    # to run once per script invocation, not once per montage.
    landmark_pos_3d, landmark_labels = load_landmarks(landmark_cache_file)

    for montage_name in montage_names:
        # Locations are hardcoded here; they must be changed as needed.
        probe_file = f"C:\\Users\\Emma\\OneDrive - NIRx Medizintechnik GmbH\\Documents\\Montage_Design\\Montages\\Aurora\\{montage_name}\\{montage_name}_probeInfo.json"
        out_file = f"C:\\Users\\Emma\\OneDrive - NIRx Medizintechnik GmbH\\Documents\\Montage_Design\\Montages\\snirf_from_probeInfo\\{montage_name}_probeInfo.snirf"

        create_probeinfo_snirf(probe_file, out_file, landmark_pos_3d, landmark_labels)