Python OIFITS module
====================

This is a Python module for reading and writing OIFITS files.  The only file you
need is oifits.py; everything else is supplementary.  For documentation on the
oifits module itself, see oifits.txt.

Note, OIFITS2 support is currently in development, although already mostly
working.  If you need OIFITS2 support, you should see the
[OIFITS2](https://github.com/pboley/oifits/tree/oifits2) branch of this
repository.

The module was tested with Astropy 4.2 and numpy 1.19.2 under Python 3.8.5.
Earlier versions will probably work, too.

For some example functions which make use of the module, see oitools.py.
These are undocumented, messy examples only and may not work; you should
probably write your own.

Also included is a sample array file for the VLTI.  This can be viewed using
the plot_array function in oitools.py:

```python
import matplotlib.pylab as plt
import oifits
from oitools import *
oidata = oifits.open('VLTI-array.fits')
plot_array(oidata.array['VLTI'])
plt.show()
```

For example usage of creating an OIFITS file from scratch, see the program
sample.py (which also requires midiwave.npz and VLTI-array.fits).

If you discover bugs, have issues with the code or suggestions, please feel
free to open an issue on Github or contact me via
email at <pboley@gmail.com>.
