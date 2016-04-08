This is a Python module for reading and writing OIFITS files.  The only file
you need is oifits.py; everything else is supplementary.  For documentation on
the oifits module itself, see oifits.txt.

The module was tested with PyFITS 3.3 and numpy 1.10.4 under both Python
2.7.10 and Python 3.4.3.  Earlier versions will probably work, too.

For some example functions which make use of the module, see oitools.py.
These are undocumented, messy examples only and may not work; you should
probably write your own.

Also included is a sample array file for the VLTI.  This can be viewed using
the plot_array function in oitools.py:

>>> import matplotlib.pylab as plt
>>> import oifits
>>> from oitools import *
>>> oidata = oifits.open('VLTI-array.fits')
>>> plot_array(oidata.array['VLTI'])
>>> plt.show()

For example usage of creating an OIFITS file from scratch, see the program
sample.py (which also requires midiwave.npz and VLTI-array.fits).
