For documentation on the oifits module itself, see oifits.txt.

The oifits module was tested with PyFITS 2.2.2 and numpy 1.3.0 under
Python 2.6.2.  Earlier versions will probably work, too.

For some example functions which make use of the module, see
oitools.py.

As people seem to have trouble saving the array/station positions
correctly (for example, all the array tables for all the OIFITS files
at http://apps.jmmc.fr/oidata/ are wrong, and in various different
ways), I've included an example for the VLTI.  This can be viewed
using the plot_array function in oitools.py:

>>> import matplotlib.pylab as plt
>>> import oifits
>>> from oitools import *
>>> oidata = oifits.open('VLTI-array.fits')
>>> plot_array(oidata.array['VLTI'])
>>> plt.show()
