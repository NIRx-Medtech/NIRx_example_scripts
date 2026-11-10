"""
Generate sensitivity profiles for fNIRS montages using the Cedalion library.
Note: Cedalion typically recommends a custom Anaconda environment, not the UV env typically used by our Support repository. See the Cedalion documentation for details.

Purpose
-------
Generate sensitivity profiles for specified fNIRS montages using a
standard head model and export the resulting visualizations as HTML files.

Inputs
------
SNIRF files containing probe information for each specified montage.

Outputs
-------
HTML files containing the exported sensitivity-profile visualizations.

Reference
---------
Cedalion image reconstruction example:
https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples/head_models/40_image_reconstruction.html

Author
------
Emma Bailey, 03.07.2026
"""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyvista as pv

import cedalion
import cedalion.data
import cedalion.dataclasses as cdc
import cedalion.dot as dot
import cedalion.io
import cedalion.sigproc.motion as motion
import cedalion.vis.anatomy
import cedalion.vis.anatomy.sensitivity_matrix
import cedalion.vis.blocks as vbx
import cedalion.xrutils as xrutils
from cedalion import units
from cedalion.io.forward_model import FluenceFile, load_Adot

xrutils.unit_stripping_is_error()

# Set this flag to True to enable interactive 3D plots
INTERACTIVE_PLOTS = True

if INTERACTIVE_PLOTS:
    pv.set_jupyter_backend('server')
else:
    pv.set_jupyter_backend('static')


def create_sensitivity_profile(probe_file, out_file, head_model):
    """Generate and export a sensitivity profile for one montage.

    Parameters
    ----------
    probe_file : str or Path
        Path to the input SNIRF probe-information file.
    out_file : str or Path
        Path for the exported HTML visualization.
    head_model : str
        Name of the standard head model to use.

    Returns
    -------
    None
        The sensitivity-profile visualization is exported to an HTML file.
    """

    # Load snirf file
    records = cedalion.io.read_snirf(probe_file)
    rec = records[0]
    geo3d_meas = rec.geo3d
    meas_list = rec._measurement_lists["amp"]

    # Load the standard head model and register the probe to its scalp.
    head_ijk = dot.get_standard_headmodel(head_model)
    geo3d_snapped_ijk = head_ijk.align_and_snap_to_scalp(geo3d_meas)

    # Build the forward model and compute the fluence
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        fwm = dot.ForwardModel(head_ijk, geo3d_snapped_ijk, meas_list)

        fluence_fname = tmp_dir_path / "fluence.h5"
        start = time.time()
        # Can't use MCX without a CUDA driver
        fwm.compute_fluence_nirfaster(fluence_fname)
        print(f"Computed fluence in {time.time() - start:.1f} s")

        # Compute the sensitivity matrix (Adot) from the fluence
        sensitivity_fname = tmp_dir_path / "sensitivity.nc"
        fwm.compute_sensitivity(fluence_fname, sensitivity_fname)
        Adot = load_Adot(sensitivity_fname)

    # Visualize the sensitivity profile on the brain/scalp surfaces
    geo3d_plot = geo3d_snapped_ijk[geo3d_snapped_ijk.type != cdc.PointType.LANDMARK]
    plotter = cedalion.vis.anatomy.sensitivity_matrix.Main(
        sensitivity=Adot,
        brain_surface=head_ijk.brain,
        head_surface=head_ijk.scalp,
        labeled_points=geo3d_plot,
    )
    plotter.plot(high_th=0, low_th=-3)

    # Cedalion's sensitivity_matrix.py sets the scalar-bar title using
    # Matplotlib mathtext syntax. Rewrite it in place before export.
    for scalar_bar in plotter.plt.scalar_bars.values():
        scalar_bar.SetTitle(scalar_bar.GetTitle().replace(r"$\log_{10}$", "log10"))

    plotter.plt.export_html(out_file)
    print(f"Saved sensitivity profile to {out_file}")


if __name__ == "__main__":
    # Choose the head model used by NIRSite
    head_model = "icbm152"

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
    
    for montage_name in montage_names:
        # Locations are hardcoded here; they must be changed as needed.
        probe_file = f"C:\\Users\\Emma\\OneDrive - NIRx Medizintechnik GmbH\\Documents\\Montage_Design\\Montages\\snirf_from_probeInfo\\{montage_name}_probeInfo.snirf"
        out_file = f"C:\\Users\\Emma\\OneDrive - NIRx Medizintechnik GmbH\\Documents\\Montage_Design\\Montages\\sensitivity_profiles_cedalion\\{montage_name}.html"

        # Generate and export the sensitivity profile for the current montage
        create_sensitivity_profile(probe_file, out_file, head_model)