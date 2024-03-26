Python OIFITS module
====================

This is a Python module for reading and writing OIFITS files.  The only file you
need is `oifits.py`; everything else is supplementary.

Note, OIFITS2 support is currently in development, although already mostly
working for both reading and writing OIFITS2 files. Please open an issue or
contact me if you find any bugs.

The module was tested with Astropy 4.2 and numpy 1.19.2 under Python 3.8.5.
Earlier versions will probably work, too.

For some example functions which make use of the module, see `oitools.py` in the
contrib directory. These are undocumented, messy examples only and may not work;
you should probably write your own.

Also included in the contrib directory is a sample array file for the VLTI.
This can be viewed using the plot_array function in `oitools.py`:

```python
import matplotlib.pylab as plt
import oifits
from oitools import plot_array
oidata = oifits.open('VLTI-array.fits')
plot_array(oidata.array['VLTI'])
plt.show()
```

If you discover bugs, have issues with the code or suggestions, please feel
free to open an issue on Github or contact me via
email at <pboley@gmail.com>.

## Installation

The module is self-contained in a single file, `oifits.py`. To use it, just put
it in the working directory of your project, or in your
[PYTHONPATH](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH)
(recommended).

## Usage

To open an existing OIFITS file, use the `oifits.open(filename)` function, where
`filename` can be either a filename or HDUList object.  This will return an
oifits object with the following members (any of which can be empty
dictionaries or numpy arrays):

- `array`: a dictionary of interferometric arrays, as defined by the OI_ARRAY
   tables.  The dictionary key is the name of the array (ARRNAME).

- `corr`: a dictionary of correlation matrices, as defined by the OI_CORR table.
  The dictionary key is the name of the table (CORRNAME).

- `header`: the header from the primary HDU of the file.

- `target`: a numpy array of targets, as defined by the rows of the OI_TARGET
table.

- `wavelength`: a dictionary of wavelength tables (OI_WAVELENGTH).  The
   dictionary key is the name of the instrument/settings (INSNAME).

- `vis`, `vis2`, `t3` and `flux`: numpy arrays of objects containing all the
   measurement information.  Each list member corresponds to a row in an
   OI_VIS/OI_VIS2/OI_T3/OI_FLUX table.

- `inspol`: a numpy array of objects containing the instrumental polarization
  information, as defined by the rows of the OI_INSPOL table.

A summary of the information in the oifits object can be obtained by
using the info() method:

```python
import oifits
oifitsobj = oifits.open('foo.fits')
oifitsobj.info()
```

The `info()` method can also be used from many of the objects contained within
the oifits object itself, e.g. `oifitsobj.array['VLTI'].info()` or
`oifitsobj.vis[0].info()`.

### Accessing the measurement data

The individual measurements can be referred to by accessing elements of the
`vis`, `vis2`, `t3` and/or `flux` numpy arrays.  These are themselves objects,
which contain the measurement data, as well as references to the corresponding
wavelength tables, intererometry stations, etc.

For example, if your OIFITS file contains OI_VIS measurements, the visibility
amplitude can be found in `oifitsobj.vis[0].visamp`, while the wavelengths
corresponding to the measurement can be found in
`oifitsobj.vis[0].visamp.wavelength.eff_wave`.

The OI_VIS/OI_VIS2/OI_T3/OI_FLUX classes use numpy masked arrays for
convenience, where the mask is defined via the `flag` member of these classes.
**Beware of the following subtlety**: the array data are accessed via (for
example) `OI_VIS.visamp`; however, `OI_VIS.visamp` itself is just a method which
constructs (on the fly) a masked array from `OI_VIS._visamp`, which is where the
data are actually stored.  This is done transparently, and the data can be
accessed and modified transparently via the "visamp" hidden attribute.  The same
goes for correlated fluxes, differential/closure phases, triple products, total
flux measurements, etc.  See the notes on the individual classes for a list of
all the "hidden" attributes.

### Combining OIFITS objects

The module provides a simple mechanism for combining multiple oifits objects,
achieved by using the `+` operator on two oifits objects: `result = a + b`.
This requires, however, that the information contained within all the support
tables (`OI_ARRAY`, `OI_WAVELENGTH`, `OI_TARGET`, etc.) is either *identical*,
or at least not mutually exclusive.

This behavior can be somewhat relaxed by setting the `matchtargetbyname` and/or
`matchstationbyname` in the top level of the module to `True`, in which cases
targets or interferometric stations with identical names will be considered as
identical (the values actually used in `result = a + b` are taken from `a`).

No mechanism is provided for the case where `OI_WAVELENGTH` tables have
identical names (INSNAME), but differing contents, as this requires
reinterpolating your data and is beyond the scope of this module.

### Creating an OIFITS object from scratch

An example of creating an OIFITS file from scratch, which simulates using the
VLTI and MIDI (now decomissioned) for observations at random hour angles of the
calibrator star HD 148478, described as a uniform disk, can be found in
`sample.py` in the contrib directory.

Once you have assembled your OIFITS object, you can check if it is *consistent*
or *valid* by using the `isconsistent()` and `isvalid()` methods of the newly
created object, respectively. *Consistency* checks whether the new object is
entirely self-contained, and does not refer to any (sub)objects (e.g. wavelength
stations, array tables, telescopes or interferometric stations) which are not
contained in the object itself. *Validity* checks whether the information
contained within the oifits object actually conforms to the OIFITS standard.  A
file can be *consistent* without being *valid*. An example of this is the
`VLTI-array.fits` file provided here, which does not conform to the OIFITS
standard in that it does not contain any measurements.  However, because the
OI_ARRAY definition is somewhat complicated and the arrays themselves don't
really change much, it can be useful to save this information separately and
reuse it as needed.

If your OIFITS object is at least *consistent*, it can be written to a FITS file
using the `save()` method.

## How to deal with non-conforming (broken) files

OIFITS files produced by the current version of the GRAVITY pipeline are known
to be broken: the `OI_ARRAY` tables are listed as conforming to OIFITS2
(`OI_REVN=2`), although they are missing the FOV and FOVTYPE columns.
Additionally, the `OI_FLUX` tables contain the flux measurements in the FLUX
column, instead of FLUXDATA. To fix the first problem, you can edit
the header of your file on-the-fly before trying to load it:

```python
from astropy.io import fits
import oifits
hdulist = fits.open('foo.fits')
hdulist['OI_ARRAY'].header['OI_REVN'] = 1
oifitsobj = oifits.open(hdulist)
```

The problem with the FLUX/FLUXDATA error is a little more tricky, so the module
handles it gracefully and warns you. Other errors in conforming to the OIFITS
standards (e.g. DATE-OBS containing a date and time, instead of just a date;
both MATISSE and GRAVITY make this mistake) are handled gracefully when
possible, however if you see warnings I would encourage you to open a bug
report with the software generating broken files.


## Documentation

The API documentation is contained within the module itself in the form of
Python docstrings, which can be accessed using Python's `help()` function.
Besides this, you should also refer to the appropriate tables in the documents
defining the standard (see [references](#references)), as the names of many
function arguments, tables, etc. are taken directly from there and not
documented explicitly.

## Acknowledgements

OIFITS2 capabilities were implemented with the financial support of grant
[18-72-10132](https://rscf.ru/en/project/18-72-10132/) of the [Russian Science
Foundation](https://rscf.ru/en/).

## References

- *A Data Exchange Standard for Optical (Visible/IR) Interferometry*; [Pauls et al., 2005](https://ui.adsabs.harvard.edu/abs/2005PASP..117.1255P/abstract)

- *OIFITS 2: the 2nd version of the data exchange standard for optical interferometry*; [Duvet et al., 2017](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A...8D/abstract)
