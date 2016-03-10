This is a Python module for reading and writing OIFITS files.  The only file
you need is oifits.py; everything else is supplementary.  For documentation on
the oifits module itself, see oifits.txt.

The module was tested with PyFITS 3.3 and numpy 1.10.4 under Python 2.7.10.
Earlier versions will probably work, too.  Python 3 is currently not
supported, but may be soon.

For some example functions which make use of the module, see oitools.py.
These are undocumented, messy examples only and may not work; you should
probably write your own.

As people seem to have trouble saving the array/station positions correctly
(for example, all the array tables for all the OIFITS files at
http://apps.jmmc.fr/oidata/ are wrong, and in various different ways), I've
included an example for the VLTI.  This can be viewed using the plot_array
function in oitools.py:

>>> import matplotlib.pylab as plt
>>> import oifits
>>> from oitools import *
>>> oidata = oifits.open('VLTI-array.fits')
>>> plot_array(oidata.array['VLTI'])
>>> plt.show()

For example usage of creating an OIFITS file from scratch, see the program
sample.py (which also requires midiwave.npz and VLTI-array.fits).
