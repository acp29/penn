# penn
Penn Lab Python 2.7 Modules for Electrophysiology Data Acquisition and Analysis

INSTALLATION

1) Place the 'penn' folder in a location in the pythonpath (e.g. path-to-python/Lib/site-packages/).

2) Move the extensions.py file from the 'penn' folder into the 'Stimfit' directory.

The 'analysis' module requires Stimfit.
The 'protocols' module requires ACQ4 (interfaced with a MultiClamp amplifier). The code in this protocol module will undoubtedly need customizing to suit your local setup.

Note that the ephysIO HDF5-based matlab file format used here for efficient
data storage is compatible with the Peaker Analysis Toolbox (Matlab Central
File Exchange #61567).

If you use code from this module, please acknowledge:
Dr Andrew Penn,
A.C.Penn@sussex.ac.uk

Sussex Neuroscience,
School of Life Sciences,
University of Sussex,
Brighton, BN1 9QG,
United Kingdom.

http://www.sussex.ac.uk/lifesci/pennlab/
