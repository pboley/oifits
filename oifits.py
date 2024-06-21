"""
A module for reading/writing OIFITS (v1, v2) files

To open an existing OIFITS file, use the oifits.open(filename) function, where
'filename' can be either a filename or HDUList object.  This will return an
oifits object with the following members (any of which can be empty
dictionaries or numpy arrays):

   array: a dictionary of interferometric arrays, as defined by the OI_ARRAY
   tables.  The dictionary key is the name of the array (ARRNAME).

   corr: a dictionary of correlation matrices, as defined by the OI_CORR table.
   The dictionary key is the name of the table (CORRNAME).

   header: the header from the primary HDU of the file.

   target: a numpy array of targets, as defined by the rows of the OI_TARGET
   table.

   wavelength: a dictionary of wavelength tables (OI_WAVELENGTH).  The
   dictionary key is the name of the instrument/settings (INSNAME).

   vis, vis2, t3 and flux: numpy arrays of objects containing all the
   measurement information.  Each list member corresponds to a row in an
   OI_VIS/OI_VIS2/OI_T3/OI_FLUX table.

   inspol: a numpy array of objects containing the instrumental polarization
   information, as defined by the rows of the OI_INSPOL table.

A summary of the information in the oifits object can be obtained by
using the info() method:

   > import oifits
   > oifitsobj = oifits.open('foo.fits')
   > oifitsobj.info()

As of version 0.4, revision 2 of the OIFITS standard (Duvert, Young & Hummel,
2017, A&A 597, A8) is supported, with the exception of correlations and
polarization.  If you need support for these features of the OIFITS2 standard,
please open an issue on github.  Support for writing OIFITS2 files is currently
experimental.

Earlier versions of this module made an ad-hoc, backwards-compatible change to
the OIFITS revision 1 standard originally described by Pauls et al., 2005, PASP,
117, 1255.  The OI_VIS tables in OIFITS files read by this module can contain
two additional columns for the correlated flux, CFLUX and CFLUXERR , which are
arrays with a length corresponding to the number of wavelength elements (just as
VISAMP). Support for writing these additional columns was removed in version
0.4, as the OIFITS standard now provides a mechanism for saving correlated flux
measurements in OI_VIS tables.

The main purpose of this module is to allow easy access to your OIFITS data
within Python, where you can then analyze it in any way you want.  As of
version 0.3, the module can now be used to create OIFITS files from scratch
without serious pain.  Be warned, creating an array table from scratch is
probably like nailing jelly to a tree.  In a future verison this may become
easier. Note that array tables are a requirement only for OIFITS2.

The module also provides a simple mechanism for combining multiple oifits
objects, achieved by using the '+' operator on two oifits objects: result = a +
b.  The result can then be written to a file using result.save(filename).

Many of the parameters and their meanings are not specifically documented here.
However, the nomenclature mirrors that of the OIFITS standard, so it is
recommended to use this module with the OIFITS1/OIFITS2 references above in
hand.

Beginning with version 0.3, the OI_VIS/OI_VIS2/OI_T3 classes now use masked
arrays for convenience, where the mask is defined via the 'flag' member of
these classes.  This also concerns the OI_FLUX tables from OIFITS2. Beware of
the following subtlety: as before, the array data are accessed via (for
example) OI_VIS.visamp; however, OI_VIS.visamp is just a method which
constructs (on the fly) a masked array from OI_VIS._visamp, which is where the
data are actually stored.  This is done transparently, and the data can be
accessed and modified transparently via the "visamp" hidden attribute.  The
same goes for correlated fluxes, differential/closure phases, triple products,
total flux measurements, etc.  See the notes on the individual classes for a
list of all the "hidden" attributes.

For further information, contact Paul Boley (pboley@gmail.com) or open an issue
on Github (https://github.com/pboley/oifits/).

"""

import numpy as np
from numpy import double, ma
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import EarthLocation
import datetime
import copy
import warnings
from packaging import version

__author__ = "Paul Boley"
__email__ = "pboley@gmail.com"
__date__ ='21 June 2024'
__version__ = '0.6-dev'
_mjdzero = datetime.datetime(1858, 11, 17)

matchtargetbyname = False
matchstationbyname = False
refdate = datetime.datetime(2000, 1, 1)

def _plurals(count):
    if count != 1: return 's'
    return ''

def _array_eq(a, b):
    "Test whether all the elements of two arrays are equal."

    if len(a) != len(b):
        return False
    try:
        return not (a != b).any()
    except:
        return not (a != b)

class _angpoint(float):
    "Convenience object for representing angles."

    def __init__(self, angle):
        self.angle = angle

    def __repr__(self):
        return '_angpoint(%s)'%self.angle.__repr__()

    def __str__(self):
        return "%g degrees"%(self.angle)

    def __eq__(self, other):
        return self.angle == other.angle

    def __ne__(self, other):
        return not self.__eq__(other)

    def asdms(self):
        """Return the value as a string in dms format,
        e.g. +25:30:22.55.  Useful for declination."""
        angle = self.angle
        if not np.isfinite(angle):
            return self.__repr__()
        if angle < 0:
            negative = True
            angle *= -1.0
        else:
            negative = False
        degrees = np.floor(angle)
        minutes = np.floor((angle - degrees)*60.0)
        seconds = (angle - degrees - minutes/60.0)*3600.0
        if negative:
            return "-%02d:%02d:%05.2f"%(degrees,minutes,seconds)
        else:
            return "+%02d:%02d:%05.2f"%(degrees,minutes,seconds)

    def ashms(self):
        """Return the value as a string in hms format,
        e.g. 5:12:17.21.  Useful for right ascension."""
        angle = self.angle*24.0/360.0
        if not np.isfinite(angle):
            return self.__repr__()
        hours = np.floor(angle)
        minutes = np.floor((angle - hours)*60.0)
        seconds = (angle - hours - minutes/60.0)*3600.0
        return "%02d:%02d:%05.2f"%(hours,minutes,seconds)

def _isnone(x):
    """Convenience hack for checking if x is none; needed because numpy
    arrays will, at some point, return arrays for x == None."""

    return type(x) == type(None)

def _notnone(x):
    """Convenience hack for checking if x is not none; needed because numpy
    arrays will, at some point, return arrays for x != None."""

    return type(x) != type(None)

class OI_TARGET(object):

    def __init__(self, target, raep0, decep0, equinox=2000.0, ra_err=0.0, dec_err=0.0,
                 sysvel=0.0, veltyp='TOPOCENT', veldef='OPTICAL', pmra=0.0, pmdec=0.0,
                 pmra_err=0.0, pmdec_err=0.0, parallax=0.0, para_err=0.0, spectyp='UNKNOWN', category=None, revision=1):

        if revision > 2:
            warnings.warn('OI_TARGET revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.target = target
        self.raep0 = _angpoint(raep0)
        self.decep0 = _angpoint(decep0)
        self.equinox = equinox
        self.ra_err = ra_err
        self.dec_err = dec_err
        self.sysvel = sysvel
        self.veltyp = veltyp
        self.veldef = veldef
        self.pmra = pmra
        self.pmdec = pmdec
        self.pmra_err = pmra_err
        self.pmdec_err = pmdec_err
        self.parallax = parallax
        self.para_err = para_err
        self.spectyp = spectyp
        if revision >= 2:
            self.category = category
        else: self.category = None


    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision  != other.revision)  or
            (self.target    != other.target)    or
            (self.raep0     != other.raep0)     or
            (self.decep0    != other.decep0)    or
            (self.equinox   != other.equinox)   or
            (self.ra_err    != other.ra_err)    or
            (self.dec_err   != other.dec_err)   or
            # Handle the case where both sysvels are nan
            ((self.sysvel != other.sysvel) != (np.isnan(self.sysvel) and np.isnan(other.sysvel))) or
            (self.veltyp    != other.veltyp)    or
            (self.veldef    != other.veldef)    or
            (self.pmra      != other.pmra)      or
            (self.pmdec     != other.pmdec)     or
            (self.pmra_err  != other.pmra_err)  or
            (self.pmdec_err != other.pmdec_err) or
            (self.parallax  != other.parallax)  or
            (self.para_err  != other.para_err)  or
            (self.spectyp   != other.spectyp)   or
            (self.category  != other.category))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        #FIXME - Add category for OIFITS2
        return "%s: %s %s (%g)"%(self.target, self.raep0.ashms(), self.decep0.asdms(), self.equinox)

    def info(self):
        print(str(self))

class OI_WAVELENGTH(object):

    def __init__(self, eff_wave, eff_band=None, revision=1):

        if revision > 2:
            warnings.warn('OI_WAVELENGTH revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.eff_wave = np.array(eff_wave, dtype=double).reshape(-1)
        if _isnone(eff_band):
            eff_band = np.zeros_like(eff_wave)
        self.eff_band = np.array(eff_band, dtype=double).reshape(-1)

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision != other.revision)              or
            (not _array_eq(self.eff_wave, other.eff_wave)) or
            (not _array_eq(self.eff_band, other.eff_band)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%d wavelength%s (%.3g-%.3g um)"%(len(self.eff_wave), _plurals(len(self.eff_wave)), 1e6*np.min(self.eff_wave),1e6*np.max(self.eff_wave))

    def info(self):
        print(str(self))


class OI_CORR(object):

    def __init__(self, iindx, jindx, corr, revision=1):

        if revision > 1:
            warnings.warn('OI_CORR revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.iindx = iindx
        self.jindx = jindx
        self.corr = corr

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision != other.revision)              or
            (not _array_eq(self.iindx, other.iindx)) or
            (not _array_eq(self.jindx, other.jindx)) or
            (not _array_eq(self.corr, other.corr)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%d correlation element%s"%(len(self.corr), _plurals(len(self.corr)))

    def info(self):
        print(str(self))

class OI_VIS(object):
    """
    Class for storing visibility amplitude and differential phase data.
    To access the data, use the following hidden attributes:

    visamp, visamperr, visphi, visphierr, flag;
    and possibly cflux, cfluxerr.

    """

    def __init__(self, timeobs, int_time, visamp, visamperr, visphi, visphierr, flag, ucoord,
                 vcoord, wavelength, target, corr=None, array=None, station=(None,None), cflux=None, cfluxerr=None, revision=1,
                 # The follow arguments are used for OIFITS2
                 corrindx_visamp=None, corrindx_visphi=None, corrindx_rvis=None, corrindx_ivis=None,
                 amptyp=None, phityp=None, amporder=None, phiorder=None,
                 ampunit=None, rvisunit=None, ivisunit=None,
                 visrefmap=None, rvis=None, rviserr=None, ivis=None, iviserr=None):

        if revision > 2:
            warnings.warn('OI_VIS revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = int_time
        self._visamp = np.array(visamp, dtype=double).reshape(-1)
        self._visamperr = np.array(visamperr, dtype=double).reshape(-1)
        self._visphi = np.array(visphi, dtype=double).reshape(-1)
        self._visphierr = np.array(visphierr, dtype=double).reshape(-1)
        if _notnone(cflux): self._cflux = np.array(cflux, dtype=double).reshape(-1)
        else: self._cflux = None
        if _notnone(cfluxerr): self._cfluxerr = np.array(cfluxerr, dtype=double).reshape(-1)
        else: self._cfluxerr = None
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.station = station
        # Only used if revision >= 2
        self.corrindx_visamp = corrindx_visamp
        self.corrindx_visphi = corrindx_visphi
        self.corrindx_rvis = corrindx_rvis
        self.corrindx_ivis = corrindx_ivis
        self.amptyp = amptyp
        self.phityp = phityp
        self.amporder = amporder
        self.phiorder = phiorder
        self.ampunit = ampunit
        self.rvisunit = rvisunit
        self.ivisunit = ivisunit
        self.visrefmap = visrefmap
        self.rvis = rvis
        self.rviserr = rviserr
        self.ivis = ivis
        self.iviserr = iviserr
        self.corr = corr

    def __eq__(self, other):

        if type(self) != type(other): return False

        # Test equality for OIFITS1
        eq = not (
            (self.revision   != other.revision)   or
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.ucoord     != other.ucoord)     or
            (self.vcoord     != other.vcoord)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self._visamp, other._visamp)) or
            (not _array_eq(self._visamperr, other._visamperr)) or
            (not _array_eq(self._visphi, other._visphi)) or
            (not _array_eq(self._visphierr, other._visphierr)) or
            (not _array_eq(self.flag, other.flag)))
        # Additional checks for OIFITS2
        if self.revision >= 2:
            eq = eq and not (
                (self.corrindx_visamp != other.corrindx_visamp) or
                (self.corrindx_visphi != other.corrindx_visphi) or
                (self.corrindx_rvis   != other.corrindx_rvis) or
                (self.corrindx_ivis   != other.corrindx_ivis) or
                (self.amptyp     != other.amptyp)    or
                (self.phityp     != other.phityp)    or
                (self.amporder   != other.amporder)  or
                (self.phiorder   != other.phiorder)  or
                (self.corr       != other.corr)      or
                (self.ampunit    != other.ampunit)   or
                (self.rvisunit   != other.rvisunit)  or
                (self.ivisunit   != other.ivisunit)  or
                (not _array_eq(self.visrefmap, other.visrefmap)) or
                (not _array_eq(self.rvis, other.rvis))           or
                (not _array_eq(self.rviserr, other.rviserr))     or
                (not _array_eq(self.ivis, other.ivis))           or
                (not _array_eq(self.iviserr, other.iviserr)))

        return eq

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('visamp', 'visamperr', 'visphi', 'visphierr'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        # Optional data arrays which may not be present, and should return None if they aren't
        elif attrname in ('cflux', 'cfluxerr'):
            if _notnone(self.__dict__['_' + attrname]):
                return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
            else:
                return None
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('visamp', 'visamperr', 'visphi', 'visphierr', 'cflux', 'cfluxerr'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meanvis = ma.mean(self.visamp)
        if self.station[0] and self.station[1]:
            baselinename = ' (' + self.station[0].sta_name + self.station[1].sta_name + ')'
        else:
            baselinename = ''
        return '%s %s%s: %d point%s (%d masked), B = %5.1f m, PA = %5.1f deg, <V> = %4.2g'%(self.target.target, self.timeobs.strftime('%F %T'), baselinename, len(self.visamp), _plurals(len(self.visamp)), np.sum(self.flag), np.sqrt(self.ucoord**2 + self.vcoord**2), np.arctan(self.ucoord / self.vcoord) * 180.0 / np.pi % 180.0, meanvis)

    def info(self):
        print(str(self))

class OI_VIS2(object):
    """
    Class for storing squared visibility amplitude data.
    To access the data, use the following hidden attributes:

    vis2data, vis2err

    """
    def __init__(self, timeobs, int_time, vis2data, vis2err, flag, ucoord, vcoord, wavelength,
                 target, corr=None, corrindx_vis2data=None, array=None, station=(None, None), revision=1):

        if revision > 2:
            warnings.warn('OI_VIS2 revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = int_time
        self._vis2data = np.array(vis2data, dtype=double).reshape(-1)
        self._vis2err = np.array(vis2err, dtype=double).reshape(-1)
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.station = station
        # Only used if revision >= 2
        self.corr = corr
        self.corrindx_vis2data = corrindx_vis2data

    def __eq__(self, other):

        if type(self) != type(other): return False

        eq = not (
            (self.revision   != other.revision)   or
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.ucoord     != other.ucoord)     or
            (self.vcoord     != other.vcoord)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self._vis2data, other._vis2data)) or
            (not _array_eq(self._vis2err, other._vis2err)) or
            (not _array_eq(self.flag, other.flag)))
        # Additional checks for OIFITS2
        if self.revision >= 2:
            eq = eq and not (
                (self.corr != other.corr))

        return eq

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('vis2data', 'vis2err'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('vis2data', 'vis2err'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meanvis = ma.mean(self.vis2data)
        if self.station[0] and self.station[1]:
            baselinename = ' (' + self.station[0].sta_name + self.station[1].sta_name + ')'
        else:
            baselinename = ''
        return "%s %s%s: %d point%s (%d masked), B = %5.1f m, PA = %5.1f deg, <V^2> = %4.2g"%(self.target.target, self.timeobs.strftime('%F %T'), baselinename, len(self.vis2data), _plurals(len(self.vis2data)), np.sum(self.flag), np.sqrt(self.ucoord**2 + self.vcoord**2), np.arctan(self.ucoord / self.vcoord) * 180.0 / np.pi % 180.0, meanvis)

    def info(self):
        print(str(self))


class OI_T3(object):
    """
    Class for storing triple product and closure phase data.
    To access the data, use the following hidden attributes:

    t3amp, t3amperr, t3phi, t3phierr

    """

    def __init__(self, timeobs, int_time, t3amp, t3amperr, t3phi, t3phierr, flag, u1coord,
                 v1coord, u2coord, v2coord, wavelength, target, corr=None,
                 corrindx_t3amp=None, corrindx_t3phi=None,
                 array=None, station=(None,None,None), revision=1):

        if revision > 2:
            warnings.warn('OI_T3 revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = int_time
        self._t3amp = np.array(t3amp, dtype=double).reshape(-1)
        self._t3amperr = np.array(t3amperr, dtype=double).reshape(-1)
        self._t3phi = np.array(t3phi, dtype=double).reshape(-1)
        self._t3phierr = np.array(t3phierr, dtype=double).reshape(-1)
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.u1coord = u1coord
        self.v1coord = v1coord
        self.u2coord = u2coord
        self.v2coord = v2coord
        self.station = station
        # Only used if revision >= 2
        self.corr = corr
        self.corrindx_t3amp = corrindx_t3amp
        self.corrindx_t3phi = corrindx_t3phi

    def __eq__(self, other):

        if type(self) != type(other): return False

        eq = not (
            (self.revision   != other.revision)   or
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.u1coord    != other.u1coord)    or
            (self.v1coord    != other.v1coord)    or
            (self.u2coord    != other.u2coord)    or
            (self.v2coord    != other.v2coord)    or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self._t3amp, other._t3amp)) or
            (not _array_eq(self._t3amperr, other._t3amperr)) or
            (not _array_eq(self._t3phi, other._t3phi)) or
            (not _array_eq(self._t3phierr, other._t3phierr)) or
            (not _array_eq(self.flag, other.flag)))
        # Additional checks for OIFITS2
        if self.revision >= 2:
            eq = eq and not (
                (self.corr != other.corr) or
                (self.corrindx_t3amp != other.corrindx_t3amp) or
                (self.corrindx_t3phi != other.corrindx_t3phi))

        return eq

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('t3amp', 't3amperr', 't3phi', 't3phierr'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('t3amp', 't3amperr', 't3phi', 't3phierr'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meant3 = np.mean(self.t3amp[np.where(self.flag == False)])
        if self.station[0] and self.station[1] and self.station[2]:
            baselinename = ' (' + self.station[0].sta_name + self.station[1].sta_name + self.station[2].sta_name + ')'
        else:
            baselinename = ''
        return "%s %s%s: %d point%s (%d masked), B = %5.1fm, %5.1fm, <T3> = %4.2g"%(self.target.target, self.timeobs.strftime('%F %T'), baselinename, len(self.t3amp), _plurals(len(self.t3amp)), np.sum(self.flag), np.sqrt(self.u1coord**2 + self.v1coord**2), np.sqrt(self.u2coord**2 + self.v2coord**2), meant3)

    def info(self):
        print(str(self))

class OI_FLUX(object):
    """
    Class for storing raw or calibrated flux measurements.
    To access the data, use the following hidden attributes:

    fluxdata, fluxerr

    """

    def __init__(self, timeobs, int_time, fluxdata, fluxerr, flag,
                 wavelength, target, calibrated, fluxunit, fluxerrunit, corr=None, array=None, station=None,
                 fov=None, fovtype=None, revision=1):

        if revision > 1:
            warnings.warn('OI_FLUX revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.corr = corr
        self.target = target
        self.int_time = int_time
        self._fluxdata = np.array(fluxdata, dtype=double).reshape(-1)
        self._fluxerr = np.array(fluxerr, dtype=double).reshape(-1)
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.station = station
        self.fov = fov
        self.fovtype = fovtype
        self.calibrated = calibrated
        self.fluxunit = fluxunit
        self.fluxerrunit = fluxerrunit

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision    != other.revision)    or
            (self.timeobs     != other.timeobs)     or
            (self.array       != other.array)       or
            (self.wavelength  != other.wavelength)  or
            (self.corr        != other.corr)        or
            (self.target      != other.target)      or
            (self.int_time    != other.int_time)    or
            (self.array       != other.array)       or
            (self.station     != other.station)     or
            (self.fov         != other.fov)         or
            (self.fovtype     != other.fovtype)     or
            (self.calibrated  != other.calibrated)  or
            (self.fluxunit    != other.fluxunit)    or
            (self.fluxerrunit != other.fluxerrunit) or
            (not _array_eq(self._fluxdata, other._fluxdata)) or
            (not _array_eq(self._fluxerr, other._fluxerr)) or
            (not _array_eq(self.flag, other.flag)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('fluxdata', 'fluxerr'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('fluxdata', 'fluxerr'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meanf = np.mean(self.fluxdata[np.where(self.flag == False)])
        if self.station:
            staname = ' (%s)'%self.station.sta_name
        else:
            staname = ''
        if self.calibrated:
            cal = 'calibrated'
        else:
            cal = 'uncalibrated'
        return "%s %s%s: %d point%s (%d masked), <F> = %4.2g (%s)"%(self.target.target, self.timeobs.strftime('%F %T'), staname, len(self.fluxdata), _plurals(len(self.fluxdata)), np.sum(self.flag), meanf, cal)

    def info(self):
        print(str(self))

class OI_STATION(object):
    """ This class corresponds to a single row (i.e. single
    station/telescope) of an OI_ARRAY table."""

    def __init__(self, tel_name=None, sta_name=None, diameter=None, staxyz=[None, None, None], fov=None, fovtype=None, revision=1):

        if revision > 2:
            warnings.warn('OI_ARRAY revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.tel_name = tel_name
        self.sta_name = sta_name
        self.diameter = diameter
        self.staxyz = staxyz

        if revision >= 2:
            self.fov = fov
            self.fovtype = fovtype
        else:
            self.fov = self.fovtype = None

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision != other.revision) or
            (self.tel_name != other.tel_name) or
            (self.sta_name != other.sta_name) or
            (self.diameter != other.diameter) or
            (not _array_eq(self.staxyz, other.staxyz)) or
            (self.fov != other.fov) or
            (self.fovtype != other.fovtype))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):

        if self.revision >= 2:
            return '%s/%s (%g m, fov %g arcsec (%s))'%(self.sta_name, self.tel_name, self.diameter, self.fov, self.fovtype)
        else:
            return '%s/%s (%g m)'%(self.sta_name, self.tel_name, self.diameter)

class OI_INSPOL(object):

    def __init__(self, timestart, timeend, orient, model, jxx, jyy, jxy, jyx, wavelength, target, array, station, revision=1):

        if revision > 1:
            warnings.warn('OI_INSPOL revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.timestart = timestart
        self.timeend = timeend
        self.orient = orient
        self.model = model
        self.jxx = np.array(jxx, dtype=complex).reshape(-1)
        self.jyy = np.array(jyy, dtype=complex).reshape(-1)
        self.jxy = np.array(jyx, dtype=complex).reshape(-1)
        self.jyx = np.array(jyx, dtype=complex).reshape(-1)
        self.wavelength = wavelength
        self.target = target
        self.array = array
        self.station = station

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision   != other.revision)   or
            (self.timestart  != other.timestart)  or
            (self.timeend    != other.timeend)    or
            (self.orient     != other.orient)     or
            (self.model      != other.model)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self.jxx, other.jxx)) or
            (not _array_eq(self.jyy, other.jyy)) or
            (not _array_eq(self.jxy, other.jxy)) or
            (not _array_eq(self.jyx, other.jyx)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):

        return '%s (%s): %s-%s'%(self.target.target, self.station.tel_name, self.timestart.strftime('%F %T'), self.timeend.strftime('%F %T'))

    def info(self):
        print(str(self))

class OI_ARRAY(object):
    """Contains all the data for a single OI_ARRAY table.  Note the
    hidden convenience attributes latitude, longitude, and altitude."""

    def __init__(self, frame, arrxyz, stations=(), revision=1):

        if revision > 2:
            warnings.warn('OI_ARRAY revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.frame = frame
        self.arrxyz = arrxyz
        self.station = np.empty(0)
        # fov/fovtype are not defined for OIFITS1; just pass them on to
        # OI_STATION constructor as None for OIFITS1.
        fov = fovtype = None
        for station in stations:
            # Go field by field, since some OIFITS files have "extra" fields
            tel_name = station['TEL_NAME']
            sta_name = station['STA_NAME']
            sta_index = station['STA_INDEX']
            diameter = station['DIAMETER']
            staxyz = station['STAXYZ']
            if revision >= 2:
                fov = station['FOV']
                fovtype = station['FOVTYPE']

            self.station = np.append(self.station, OI_STATION(tel_name=tel_name, sta_name=sta_name, diameter=diameter, staxyz=staxyz, fov=fov, fovtype=fovtype, revision=revision))

    def __eq__(self, other):

        if type(self) != type(other): return False

        equal = not (
            (self.revision != other.revision) or
            (self.frame   != other.frame)   or
            (not _array_eq(self.arrxyz, other.arrxyz)))

        if not equal: return False

        # If position appears to be the same, check that the stations
        # (and ordering) are also the same
        if (self.station != other.station).any():
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname == 'latitude':
            if self.frame == 'GEOCENTRIC':
                c = EarthLocation(*self.arrxyz*u.m)
                return _angpoint(c.lat.value)
            else:
                warnings.warn('Latitude only defined for geocentric coordinates', UserWarning)
                return _angpoint(np.nan)
        elif attrname == 'longitude':
            if self.frame == 'GEOCENTRIC':
                c = EarthLocation(*self.arrxyz*u.m)
                return _angpoint(c.lon.value)
            else:
                warnings.warn('Longitude only defined for geocentric coordinates', UserWarning)
                return _angpoint(np.nan)
        elif attrname == 'altitude':
            if self.frame == 'GEOCENTRIC':
                c = EarthLocation(*self.arrxyz*u.m)
                return c.height.value
            else:
                warnings.warn('Height only defined for geocentric coordinates', UserWarning)
                return np.nan
        else:
            raise AttributeError(attrname)

    def __repr__(self):
        # FIXME -- add frame
        return '%s %s %g m, %d station%s'%(self.latitude.asdms(), self.longitude.asdms(), self.altitude, len(self.station), _plurals(len(self.station)))

    def info(self, verbose=0):
        """Print the array's center coordinates.  If verbosity >= 1,
        print information about each station."""
        print(str(self))
        if verbose >= 1:
            for station in self.station:
                print("   %s"%str(station))

    def get_station_by_name(self, name):

        for station in self.station:
            if station.sta_name == name:
                return station

        raise LookupError('No such station %s'%name)

class oifits(object):

    def __init__(self):

        self.header = None
        self.wavelength = {}
        self.corr = {}
        self.target = np.empty(0)
        self.array = {}
        self.vis = np.empty(0)
        self.vis2 = np.empty(0)
        self.t3 = np.empty(0)
        self.flux = np.empty(0)
        self.inspol = np.empty(0)

    def __add__(self, other):
        """Consistently combine two separate oifits objects.  Note
        that targets can be matched by name only (e.g. if coordinates
        differ) by setting oifits.matchtargetbyname to True.  The same
        goes for stations of the array (controlled by
        oifits.matchstationbyname)"""
        # Don't do anything if the two oifits objects are not CONSISTENT!
        if self.isconsistent() == False or other.isconsistent() == False:
            raise ValueError('oifits objects are not consistent, bailing')

        new = copy.deepcopy(self)

        if new.header != None:
            if other.header != None:
                # Older versions of pyfits don't allow combining headers
                try:
                    new.header = new.header + other.header
                except TypeError:
                    warnings.warn('Warning: Keeping FITS header from first oifits object', UserWarning)
        elif other.header != None:
            new.header = other.header.copy()

        if len(other.wavelength):
            wavelengthmap = {}
            for key in other.wavelength.keys():
                if key not in new.wavelength.keys():
                    new.wavelength[key] = copy.deepcopy(other.wavelength[key])
                elif new.wavelength[key] != other.wavelength[key]:
                    raise ValueError('Wavelength tables have the same key but differing contents.')
                wavelengthmap[id(other.wavelength[key])] = new.wavelength[key]

        if len(other.corr):
            corrmap = {}
            for key in other.corr.keys():
                if key not in new.corr.keys():
                    new.corr[key] = copy.deepcopy(other.corr[key])
                elif new.corr[key] != other.corr[key]:
                    raise ValueError('Correlation matrices have the same key but differing contents.')
                corrmap[id(other.corr[key])] = new.corr[key]

        if len(other.target):
            targetmap = {}
            for otarget in other.target:
                for ntarget in new.target:
                    if matchtargetbyname and ntarget.target == otarget.target:
                        targetmap[id(otarget)] = ntarget
                        break
                    elif ntarget == otarget:
                        targetmap[id(otarget)] = ntarget
                        break
                    elif ntarget.target == otarget.target:
                        print('Found a target with a matching name, but some differences in the target specification.  Creating a new target.  Set oifits.matchtargetbyname to True to override this behavior.')
                # If 'id(otarget)' is not in targetmap, then this is a new
                # target and should be added to the array of targets
                if id(otarget) not in targetmap.keys():
                    try:
                        newkey = new.target.keys()[-1]+1
                    except:
                        newkey = 1
                    target = copy.deepcopy(otarget)
                    new.target = np.append(new.target, target)
                    targetmap[id(otarget)] = target

        if len(other.array):
            stationmap = {}
            arraymap = {}
            for key, otharray in other.array.items():
                arraymap[id(otharray)] = key
                if key not in new.array.keys():
                    new.array[key] = copy.deepcopy(other.array[key])
                # If arrays have the same name but seem to differ, try
                # to combine the two (by including the union of both
                # sets of stations)
                for othsta in other.array[key].station:
                    for newsta in new.array[key].station:
                        if newsta == othsta:
                            stationmap[id(othsta)] = newsta
                            break
                        elif matchstationbyname and newsta.sta_name == othsta.sta_name:
                            stationmap[id(othsta)] = newsta
                            break
                        elif newsta.sta_name == othsta.sta_name and matchstationbyname == False:
                            raise ValueError('Stations have matching names but conflicting data.')
                    # If 'id(othsta)' is not in the stationmap
                    # dictionary, then this is a new station and
                    # should be added to the current array
                    if id(othsta) not in stationmap.keys():
                        newsta = copy.deepcopy(othsta)
                        new.array[key].station = np.append(new.array[key].station, newsta)
                        stationmap[id(othsta)] = newsta
                        # Make sure that staxyz of the new station is relative to the new array center
                        newsta.staxyz = othsta.staxyz - other.array[key].arrxyz + new.array[key].arrxyz

        for vis in other.vis:
            if vis not in new.vis:
                newvis = copy.copy(vis)
                # The wavelength, target, corr (if present), array and station
                # objects should point to the appropriate objects inside the
                # 'new' structure
                newvis.wavelength = wavelengthmap[id(vis.wavelength)]
                newvis.target = targetmap[id(vis.target)]
                if (vis.corr):
                    newvis.corr = corrmap[id(vis.corr)]
                if (vis.array):
                    newvis.array = new.array[arraymap[id(vis.array)]]
                    newvis.station = [None, None]
                    newvis.station[0] = stationmap[id(vis.station[0])]
                    newvis.station[1] = stationmap[id(vis.station[1])]
                new.vis = np.append(new.vis, newvis)

        for vis2 in other.vis2:
            if vis2 not in new.vis2:
                newvis2 = copy.copy(vis2)
                # The wavelength, target, corr (if present), array and station
                # objects should point to the appropriate objects inside the
                # 'new' structure
                newvis2.wavelength = wavelengthmap[id(vis2.wavelength)]
                newvis2.target = targetmap[id(vis2.target)]
                if (vis2.corr):
                    newvis2.corr = corrmap[id(vis2.corr)]
                if (vis2.array):
                    newvis2.array = new.array[arraymap[id(vis2.array)]]
                    newvis2.station = [None, None]
                    newvis2.station[0] = stationmap[id(vis2.station[0])]
                    newvis2.station[1] = stationmap[id(vis2.station[1])]
                new.vis2 = np.append(new.vis2, newvis2)

        for t3 in other.t3:
            if t3 not in new.t3:
                newt3 = copy.copy(t3)
                # The wavelength, target, corr (if present), array and station
                # objects should point to the appropriate objects inside the
                # 'new' structure
                newt3.wavelength = wavelengthmap[id(t3.wavelength)]
                newt3.target = targetmap[id(t3.target)]
                if (t3.corr):
                    newt3.corr = corrmap[id(t3.corr)]
                if (t3.array):
                    newt3.array = new.array[arraymap[id(t3.array)]]
                    newt3.station = [None, None, None]
                    newt3.station[0] = stationmap[id(t3.station[0])]
                    newt3.station[1] = stationmap[id(t3.station[1])]
                    newt3.station[2] = stationmap[id(t3.station[2])]
                new.t3 = np.append(new.t3, newt3)

        for flux in other.flux:
            if flux not in new.flux:
                newflux = copy.copy(flux)
                # The wavelength, target, corr (if present), array and station
                # objects should point to the appropriate objects inside the
                # 'new' structure
                newflux.wavelength = wavelengthmap[id(flux.wavelength)]
                newflux.target = targetmap[id(flux.target)]
                if (flux.corr):
                    newflux.corr = corrmap[id(flux.corr)]
                if (flux.array):
                    newflux.array = new.array[arraymap[id(flux.array)]]
                    newflux.station = stationmap[id(flux.station)]
                new.flux = np.append(new.flux, newflux)

        for inspol in other.inspol:
                newinspol = copy.copy(inspol)
                # The wavelength, target, corr (if present), array and station
                # objects should point to the appropriate objects inside the
                # 'new' structure
                newinspol.wavelength = wavelengthmap[id(inspol.wavelength)]
                newinspol.target = targetmap[id(inspol.target)]
                newinspol.array = new.array[arraymap[id(inspol.array)]]
                newinspol.station = stationmap[id(inspol.station)]
                new.inspol = np.append(new.inspol, newinspol)


        return(new)


    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.wavelength != other.wavelength)   or
            (self.corr       != other.corr)         or
            (self.target     != other.target).any() or
            (self.array      != other.array)      or
            (self.vis        != other.vis).any()  or
            (self.vis2       != other.vis2).any() or
            (self.t3         != other.t3).any()   or
            (self.flux       != other.flux).any() or
            (self.inspol     != other.inspol).any())

    def __ne__(self, other):
        return not self.__eq__(other)

    def isvalid(self):
        """Returns True if the oifits object is both consistent (as
        determined by isconsistent()) and conforms to the OIFITS
        standard (according to Pauls et al., 2005, PASP, 117, 1255)."""

        warnings = []
        errors = []
        if not self.isconsistent():
            errors.append('oifits object is not consistent')
        if not self.target.size:
            errors.append('No OI_TARGET data')
        if not self.wavelength:
            errors.append('No OI_WAVELENGTH data')
        else:
            for key, wavelength in self.wavelength.items():
                if len(wavelength.eff_wave) != len(wavelength.eff_band):
                    errors.append("eff_wave and eff_band are of different lengths for wavelength table '%s'"%key)
        for key, corr in self.corr.items():
            ndata = len(corr.iindx)
            if (len(corr.jindx) != ndata) or (len(corr.corr) != ndata):
                errors.append("Number of indices/elements not consistent in correlation table '%s'"%key)
        if (self.vis.size + self.vis2.size + self.t3.size + self.flux.size == 0):
            errors.append('Need to have atleast one measurement table (vis, vis2, t3, flux)')
        for vis in self.vis:
            nwave = len(vis.wavelength.eff_band)
            if (len(vis.visamp) != nwave) or (len(vis.visamperr) != nwave) or (len(vis.visphi) != nwave) or (len(vis.visphierr) != nwave) or (len(vis.flag) != nwave):
                errors.append("Data size mismatch for visibility measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))
        for vis2 in self.vis2:
            nwave = len(vis2.wavelength.eff_band)
            if (len(vis2.vis2data) != nwave) or (len(vis2.vis2err) != nwave) or (len(vis2.flag) != nwave):
                errors.append("Data size mismatch for visibility^2 measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))
        for t3 in self.t3:
            nwave = len(t3.wavelength.eff_band)
            if (len(t3.t3amp) != nwave) or (len(t3.t3amperr) != nwave) or (len(t3.t3phi) != nwave) or (len(t3.t3phierr) != nwave) or (len(t3.flag) != nwave):
                errors.append("Data size mismatch for t3 measurement 0x%x (wavelength table has a length of %d)"%(id(t3), nwave))
        for flux in self.flux:
            nwave = len(flux.wavelength.eff_band)
            if (len(flux.fluxdata) != nwave) or (len(flux.fluxerr) != nwave) or (len(flux.flag) != nwave):
                errors.append("Data size mismatch for flux measurement 0x%x (wavelength table has a length of %d)"%(id(flux), nwave))
        for inspol in self.inspol:
            nwave = len(inspol.wavelength.eff_band)
            if (len(inspol.jxx) != nwave) or (len(inspol.jyy) != nwave) or (len(inspol.jxy) != nwave) or (len(inspol.jyx) != nwave):
                errors.append("Data size mismatch for inspol measurement 0x%x (wavelength table has a length of %d)"%(id(flux), nwave))

        if warnings:
            print("*** %d warning%s:"%(len(warnings), _plurals(len(warnings))))
            for warning in warnings:
                print('  ' + warning)
        if errors:
            print("*** %d ERROR%s:"%(len(errors), _plurals(len(errors)).upper()))
            for error in errors:
                print('  ' + error)

        return not (len(warnings) or len(errors))

    def isconsistent(self):
        """Returns True if the object is entirely self-contained,
        i.e. all cross-references to wavelength tables, arrays,
        stations etc. in the measurements refer to elements which are
        stored in the oifits object.  Note that an oifits object can
        be 'consistent' in this sense without being 'valid' as checked
        by isvalid()."""

        for vis in self.vis:
            if vis.array and (vis.array not in self.array.values()):
                print('A visibility measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(vis))
                return False
            if ((vis.station[0] and (vis.station[0] not in vis.array.station)) or
                (vis.station[1] and (vis.station[1] not in vis.array.station))):
                print('A visibility measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(vis))
                return False
            if vis.wavelength not in self.wavelength.values():
                print('A visibility measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(vis))
                return False
            if vis.revision >= 2 and vis.corr and (vis.corr not in self.corr.values()):
                print('A visibility measurement (0x%x) refers to a correlation table which is not inside the main oifits object.'%id(vis))
                return False
            if vis.target not in self.target:
                print('A visibility measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(vis))
                return False

        for vis2 in self.vis2:
            if vis2.array and (vis2.array not in self.array.values()):
                print('A visibility^2 measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(vis2))
                return False
            if ((vis2.station[0] and (vis2.station[0] not in vis2.array.station)) or
                (vis2.station[1] and (vis2.station[1] not in vis2.array.station))):
                print('A visibility^2 measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(vis))
                return False
            if vis2.wavelength not in self.wavelength.values():
                print('A visibility^2 measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(vis2))
                return False
            if vis2.revision >= 2 and vis2.corr and (vis2.corr not in self.corr.values()):
                print('A visibility^2 measurement (0x%x) refers to a correlation table which is not inside the main oifits object.'%id(vis2))
                return False
            if vis2.target not in self.target:
                print('A visibility^2 measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(vis2))
                return False

        for t3 in self.t3:
            if t3.array and (t3.array not in self.array.values()):
                print('A closure phase measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(t3))
                return False
            if ((t3.station[0] and (t3.station[0] not in t3.array.station)) or
                (t3.station[1] and (t3.station[1] not in t3.array.station)) or
                (t3.station[2] and (t3.station[2] not in t3.array.station))):
                print('A closure phase measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(t3))
                return False
            if t3.wavelength not in self.wavelength.values():
                print('A closure phase measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(t3))
                return False
            if t3.revision >= 2 and t3.corr and (t3.corr not in self.corr.values()):
                print('A closure phase measurement (0x%x) refers to a correlation table which is not inside the main oifits object.'%id(t3))
                return False
            if t3.target not in self.target:
                print('A closure phase measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(t3))
                return False

        for flux in self.flux:
            if flux.array and (flux.array not in self.array.values()):
                print('A flux measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(flux))
                return False
            if flux.station and (flux.station not in flux.array.station):
                print('A flux measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(flux))
                return False
            if flux.wavelength not in self.wavelength.values():
                print('A flux measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(flux))
                return False
            if flux.corr and (flux.corr not in self.corr.values()):
                print('A flux measurement (0x%x) refers to a correlation table which is not inside the main oifits object.'%id(flux))
                return False
            if flux.target not in self.target:
                print('A flux measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(flux))
                return False

        for inspol in self.inspol:
            if inspol.array not in self.array.values():
                print('An inspol measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(inspol))
                return False
            if inspol.station not in inspol.array.station:
                print('An inspol measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(inspol))
                return False
            if inspol.wavelength not in self.wavelength.values():
                print('An inspol measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(inspol))
                return False
            if inspol.target not in self.target:
                print('An inspol measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(inspol))
                return False

        return True

    def getoifitsver(self):
        """Get the minimum OIFITS "version" of the object.  This is based on
        revision numbers of the individual tables, and the presence or absence
        of some tables (e.g. OI_INSPOL, OI_CORR, OI_FLUX.

        As of now (Jan 2021) returns only 1 or 2. A version of "1" means there
        are no OIFITS2 tables present; a verision of "2" means there is at
        least one OIFITS2 table present."""

        for wavelength in self.wavelength.values():
            if wavelength.revision >= 2:
                return 2

        for target in self.target:
            if target.revision >= 2:
                return 2

        for array in self.array.values():
            if array.revision >= 2:
                return 2

        for vis in self.vis:
            if vis.revision >= 2:
                return 2

        for vis2 in self.vis2:
            if vis2.revision >= 2:
                return 2

        for t3 in self.t3:
            if t3.revision >= 2:
                return 2

        if len(self.corr) or len(self.flux) or len(self.inspol):
            return 2

        return 1

    def info(self, recursive=True, verbose=0):
        """Print out a summary of the contents of the oifits object.
        Set recursive=True to obtain more specific information about
        each of the individual components, and verbose to an integer
        to increase the verbosity level."""

        if self.wavelength:
            wavelengths = 0
            if recursive:
                print("====================================================================")
                print("SUMMARY OF WAVELENGTH TABLES")
                print("====================================================================")
            for key in self.wavelength.keys():
                wavelengths += len(self.wavelength[key].eff_wave)
                if recursive: print("'%s': %s"%(key, str(self.wavelength[key])))
            print("%d wavelength table%s with %d wavelength%s in total"%(len(self.wavelength), _plurals(len(self.wavelength)), wavelengths, _plurals(wavelengths)))
        if self.corr:
            corrs = 0
            if recursive:
                print("====================================================================")
                print("SUMMARY OF CORRELATION TABLES")
                print("====================================================================")
            for key in self.corr.keys():
                corrs += len(self.corr[key].corr)
                if recursive: print("'%s': %s"%(key, str(self.corr[key])))
            print("%d correlation table%s with %d matrix element%s in total"%(len(self.corr), _plurals(len(self.corr)), corrs, _plurals(corrs)))
        if self.target.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF TARGET TABLES")
                print("====================================================================")
                for target in self.target:
                    target.info()
            print("%d target%s"%(len(self.target), _plurals(len(self.target))))
        if self.array:
            stations = 0
            if recursive:
                print("====================================================================")
                print("SUMMARY OF ARRAY TABLES")
                print("====================================================================")
            for key in self.array.keys():
                if recursive:
                    print(key + ':')
                    self.array[key].info(verbose=verbose)
                stations += len(self.array[key].station)
            print("%d array%s with %d station%s"%(len(self.array), _plurals(len(self.array)), stations, _plurals(stations)))
        if self.vis.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF VISIBILITY MEASUREMENTS")
                print("====================================================================")
                for vis in self.vis:
                    vis.info()
            print("%d visibility measurement%s"%(len(self.vis), _plurals(len(self.vis))))
        if self.vis2.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF VISIBILITY^2 MEASUREMENTS")
                print("====================================================================")
                for vis2 in self.vis2:
                    vis2.info()
            print("%d visibility^2 measurement%s"%(len(self.vis2), _plurals(len(self.vis2))))
        if self.t3.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF T3 MEASUREMENTS")
                print("====================================================================")
                for t3 in self.t3:
                    t3.info()
            print("%d closure phase measurement%s"%(len(self.t3), _plurals(len(self.t3))))
        if self.flux.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF FLUX MEASUREMENTS")
                print("====================================================================")
                for flux in self.flux:
                    flux.info()
            print("%d flux measurement%s"%(len(self.flux), _plurals(len(self.flux))))
        if self.inspol.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF INSPOL MEASUREMENTS")
                print("====================================================================")
                for inspol in self.inspol:
                    inspol.info()
            print("%d inspol measurement%s"%(len(self.inspol), _plurals(len(self.inspol))))

    def save(self, filename, overwrite=False):
        """Write the contents of the oifits object to a file in OIFITS
        format."""

        if not self.isconsistent():
            raise ValueError('oifits object is not consistent; refusing to go further')

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(header=self.header)
        hdu.header['DATE'] = datetime.datetime.now().strftime(format='%F'), 'Creation date'
        # Remove old oifits.py comments if they are present
        remcomments = []
        try:
            for i, comment in enumerate(hdu.header['COMMENT']):
                if (('Written by OIFITS Python module' in str(comment)) |
                    ('http://www.mpia-hd.mpg.de/homes/boley/oifits/' in str(comment)) |
                    ('http://astro.ins.urfu.ru/pages/~pboley/oifits/' in str(comment))):
                    remcomments.append(i)
        except KeyError:
            # KeyError should be raised if there are no comments
            pass
        # Cards should be removed from the bottom, otherwise the
        # ordering can get messed up and header.ascard.remove can fail
        remcomments.reverse()
        for i in remcomments:
            del hdu.header[('COMMENT', i)]
        # Add (new) advertisement
        hdu.header.add_comment('Written by OIFITS Python module version %s'%__version__)
        hdu.header.add_comment('https://github.com/pboley/oifits')
        hdulist.append(hdu)

        wavelengthmap = {}
        for insname, wavelength in self.wavelength.items():
            wavelengthmap[id(wavelength)] = insname
            hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
                fits.Column(name='EFF_WAVE', format='1E', unit='METERS', array=wavelength.eff_wave),
                fits.Column(name='EFF_BAND', format='1E', unit='METERS', array=wavelength.eff_band)
                )))
            hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
            hdu.header['OI_REVN'] = wavelength.revision, 'Revision number of the table definition'
            hdu.header['INSNAME'] = insname, 'Name of detector, for cross-referencing'
            hdulist.append(hdu)

        corrmap = {}
        for corrname, corr in self.corr.items():
            corrmap[id(corr)] = corrname
            hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
                fits.Column(name='IINDX', format='1J', array=corr.iindx),
                fits.Column(name='JINDX', format='1J', array=corr.iindx),
                fits.Column(name='CORR', format='D1', array=corr.corr)
                )))
            hdu.header['EXTNAME'] = 'OI_CORR'
            hdu.header['OI_REVN'] = corr.revision, 'Revision number of the table definition'
            hdu.header['CORRNAME'] = corrname
            hdulist.append(hdu)

        targetmap = {}
        if self.target.size:
            target_id = []
            target = []
            raep0 = []
            decep0 = []
            equinox = []
            ra_err = []
            dec_err = []
            sysvel = []
            veltyp = []
            veldef = []
            pmra = []
            pmdec = []
            pmra_err = []
            pmdec_err = []
            parallax = []
            para_err = []
            spectyp = []
            category = []
            revision = 1
            for i, targ in enumerate(self.target):
                key = i+1
                targetmap[id(targ)] = key
                target_id.append(key)
                target.append(targ.target)
                raep0.append(targ.raep0)
                decep0.append(targ.decep0)
                equinox.append(targ.equinox)
                ra_err.append(targ.ra_err)
                dec_err.append(targ.dec_err)
                sysvel.append(targ.sysvel)
                veltyp.append(targ.veltyp)
                veldef.append(targ.veldef)
                pmra.append(targ.pmra)
                pmdec.append(targ.pmdec)
                pmra_err.append(targ.pmra_err)
                pmdec_err.append(targ.pmdec_err)
                parallax.append(targ.parallax)
                para_err.append(targ.para_err)
                spectyp.append(targ.spectyp)
                category.append(targ.category or '') # Replace None with empty string
                # Check if any of the targets are higher than revision 1; save
                # everything with highest revision used
                if targ.revision > revision: revision = targ.revision


            cols = [fits.Column(name='TARGET_ID', format='1I', array=target_id),
                    fits.Column(name='TARGET', format='16A', array=target),
                    fits.Column(name='RAEP0', format='1D', unit='DEGREES', array=raep0),
                    fits.Column(name='DECEP0', format='1D', unit='DEGREES', array=decep0),
                    fits.Column(name='EQUINOX', format='1E', unit='YEARS', array=equinox),
                    fits.Column(name='RA_ERR', format='1D', unit='DEGREES', array=ra_err),
                    fits.Column(name='DEC_ERR', format='1D', unit='DEGREES', array=dec_err),
                    fits.Column(name='SYSVEL', format='1D', unit='M/S', array=sysvel),
                    fits.Column(name='VELTYP', format='8A', array=veltyp),
                    fits.Column(name='VELDEF', format='8A', array=veldef),
                    fits.Column(name='PMRA', format='1D', unit='DEG/YR', array=pmra),
                    fits.Column(name='PMDEC', format='1D', unit='DEG/YR', array=pmdec),
                    fits.Column(name='PMRA_ERR', format='1D', unit='DEG/YR', array=pmra_err),
                    fits.Column(name='PMDEC_ERR', format='1D', unit='DEG/YR', array=pmdec_err),
                    fits.Column(name='PARALLAX', format='1E', unit='DEGREES', array=parallax),
                    fits.Column(name='PARA_ERR', format='1E', unit='DEGREES', array=para_err),
                    fits.Column(name='SPECTYP', format='16A', array=spectyp)]
            if revision >= 2:
                cols.append(fits.Column(name='CATEGORY', format='3A', array=category))

            hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
            hdu.header['EXTNAME'] = 'OI_TARGET'
            hdu.header['OI_REVN'] = revision, 'Revision number of the table definition'
            hdulist.append(hdu)

        arraymap = {}
        stationmap = {}
        revision = 1
        # Check if any of the arrays or stations are higher than revision 1;
        # save everything with highest revision used
        for array in self.array.values():
            if array.revision > revision: revision = array.revision
            for station in array.station:
                if station.revision > revision: revision = station.revision
        for arrname, array in self.array.items():
            arraymap[id(array)] = arrname
            tel_name = []
            sta_name = []
            sta_index = []
            diameter = []
            staxyz = []
            fov = []
            fovtype = []
            if array.station.size:
                for i, station in enumerate(array.station, 1):
                    stationmap[id(station)] = i
                    tel_name.append(station.tel_name)
                    sta_name.append(station.sta_name)
                    sta_index.append(i)
                    diameter.append(station.diameter)
                    staxyz.append(station.staxyz)
                    fov.append(station.fov or 0) # Replace None with 0
                    fovtype.append(station.fovtype or 'UNDEF') # Replace None with UNDEF
                cols = [fits.Column(name='TEL_NAME', format='16A', array=tel_name),
                        fits.Column(name='STA_NAME', format='16A', array=sta_name),
                        fits.Column(name='STA_INDEX', format='1I', array=sta_index),
                        fits.Column(name='DIAMETER', unit='METERS', format='1E', array=diameter),
                        fits.Column(name='STAXYZ', unit='METERS', format='3D', array=staxyz)]
                if revision >= 2:
                    cols.append(fits.Column(name='FOV', format='D1', array=fov))
                    cols.append(fits.Column(name='FOVTYPE', format='A6', array=fovtype))
                hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
            hdu.header['EXTNAME'] = 'OI_ARRAY'
            hdu.header['OI_REVN'] = revision, 'Revision number of the table definition'
            hdu.header['ARRNAME'] = arrname, 'Array name, for cross-referencing'
            hdu.header['FRAME'] = array.frame, 'Coordinate frame'
            hdu.header['ARRAYX'] = array.arrxyz[0], 'Array center x coordinate (m)'
            hdu.header['ARRAYY'] = array.arrxyz[1], 'Array center y coordinate (m)'
            hdu.header['ARRAYZ'] = array.arrxyz[2], 'Array center z coordinate (m)'
            hdulist.append(hdu)

        if self.vis.size:
            # The tables are grouped by ARRNAME and INSNAME -- all
            # observations which have the same ARRNAME and INSNAME are
            # put into a single FITS binary table.
            tables = {}
            # Check if any of the vis tables are higher than revision 1; save
            # everything with highest revision used
            revision = 1
            for vis in self.vis:
                if vis.revision > revision: revision = vis.revision
            for vis in self.vis:
                nwave = vis.wavelength.eff_wave.size
                key = (arraymap.get(id(vis.array)), wavelengthmap.get(id(vis.wavelength)), corrmap.get(id(vis.corr)), vis.amptyp, vis.phityp)
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          'visamp':[], 'visamperr':[], 'visphi':[], 'visphierr':[],
                                          'ucoord':[], 'vcoord':[],
                                          'sta_index':[], 'flag':[]}
                if _notnone(vis.cflux) or _notnone(vis.cfluxerr):
                    warnings.warn('CFLUX columns in OI_VIS object will not be saved.', UserWarning)
                data['target_id'].append(targetmap[id(vis.target)])
                if vis.timeobs:
                    time = vis.timeobs - refdate
                    data['time'].append(time.days * 24.0 * 3600.0 + time.seconds)
                    mjd = (vis.timeobs - _mjdzero).days + (vis.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                    data['mjd'].append(mjd)
                else:
                    data['time'].append(None)
                    data['mjd'].append(None)
                data['int_time'].append(vis.int_time)
                if nwave == 1:
                    data['visamp'].append(vis.visamp[0])
                    data['visamperr'].append(vis.visamperr[0])
                    data['visphi'].append(vis.visphi[0])
                    data['visphierr'].append(vis.visphierr[0])
                    data['flag'].append(vis.flag[0])
                else:
                    data['visamp'].append(vis.visamp)
                    data['visamperr'].append(vis.visamperr)
                    data['visphi'].append(vis.visphi)
                    data['visphierr'].append(vis.visphierr)
                    data['flag'].append(vis.flag)
                data['ucoord'].append(vis.ucoord)
                data['vcoord'].append(vis.vcoord)
                if vis.station[0] and vis.station[1]:
                    data['sta_index'].append([stationmap[id(vis.station[0])], stationmap[id(vis.station[1])]])
                else:
                    data['sta_index'].append([-1, -1])
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size
                cols = [fits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                        fits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                        fits.Column(name='MJD', unit='DAY', format='1D', array=data['mjd']),
                        fits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time'])]
                # If TUNITs should be specified, do so
                if (revision >= 2) and (key[3] == 'correlated flux'):
                    cols += [fits.Column(name='VISAMP', unit=vis.ampunit, format='%dD'%nwave, array=data['visamp']),
                             fits.Column(name='VISAMPERR', unit=vis.ampunit, format='%dD'%nwave, array=data['visamperr'])]
                else:
                    cols += [fits.Column(name='VISAMP', format='%dD'%nwave, array=data['visamp']),
                             fits.Column(name='VISAMPERR', format='%dD'%nwave, array=data['visamperr'])]
                cols += [fits.Column(name='VISPHI', unit='DEGREES', format='%dD'%nwave, array=data['visphi']),
                         fits.Column(name='VISPHIERR', unit='DEGREES', format='%dD'%nwave, array=data['visphierr']),
                         fits.Column(name='UCOORD', format='1D', unit='METERS', array=data['ucoord']),
                         fits.Column(name='VCOORD', format='1D', unit='METERS', array=data['vcoord']),
                         fits.Column(name='STA_INDEX', format='2I', array=data['sta_index'], null=-1),
                         fits.Column(name='FLAG', format='%dL'%nwave)]
                hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))

                # Setting the data of logical field via the
                # fits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header['EXTNAME'] = 'OI_VIS'
                hdu.header['OI_REVN'] = revision, 'Revision number of the table definition'
                hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
                if key[0]: hdu.header['ARRNAME'] = key[0], 'Identifies corresponding OI_ARRAY'
                hdu.header['INSNAME'] = key[1], 'Identifies corresponding OI_WAVELENGTH table'
                if key[2]: hdu.header['CORRNAME'] = key[2], 'Identifies corresponding OI_CORR table'
                if key[3]: hdu.header['AMPTYP'] = key[3], 'Type for amplitude measurement'
                if key[4]: hdu.header['PHITYP'] = key[4], 'Type for phi measurement'
                hdulist.append(hdu)

        if self.vis2.size:
            tables = {}
            # Check if any of the vis2 tables are higher than revision 1; save
            # everything with highest revision used
            revision = 1
            for vis in self.vis2:
                if vis.revision > revision: revision = vis.revision
            for vis in self.vis2:
                nwave = vis.wavelength.eff_wave.size
                key = (arraymap.get(id(vis.array)), wavelengthmap.get(id(vis.wavelength)), corrmap.get(id(vis.corr)))
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          'vis2data':[], 'vis2err':[], 'ucoord':[], 'vcoord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(targetmap[id(vis.target)])
                if vis.timeobs:
                    time = vis.timeobs - refdate
                    data['time'].append(time.days * 24.0 * 3600.0 + time.seconds)
                    mjd = (vis.timeobs - _mjdzero).days + (vis.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                    data['mjd'].append(mjd)
                else:
                    data['time'].append(None)
                    data['mjd'].append(None)
                data['int_time'].append(vis.int_time)
                if nwave == 1:
                    data['vis2data'].append(vis.vis2data[0])
                    data['vis2err'].append(vis.vis2err[0])
                    data['flag'].append(vis.flag[0])
                else:
                    data['vis2data'].append(vis.vis2data)
                    data['vis2err'].append(vis.vis2err)
                    data['flag'].append(vis.flag)
                data['ucoord'].append(vis.ucoord)
                data['vcoord'].append(vis.vcoord)
                if vis.station[0] and vis.station[1]:
                    data['sta_index'].append([stationmap[id(vis.station[0])], stationmap[id(vis.station[1])]])
                else:
                    data['sta_index'].append([-1, -1])
                if vis.corr:
                    raise NotImplementedError('Writing correlation information from OI_VIS2 tables is not yet implemented')
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
                    fits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    fits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    fits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                    fits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    fits.Column(name='VIS2DATA', format='%dD'%nwave, array=data['vis2data']),
                    fits.Column(name='VIS2ERR', format='%dD'%nwave, array=data['vis2err']),
                    fits.Column(name='UCOORD', format='1D', unit='METERS', array=data['ucoord']),
                    fits.Column(name='VCOORD', format='1D', unit='METERS', array=data['vcoord']),
                    fits.Column(name='STA_INDEX', format='2I', array=data['sta_index'], null=-1),
                    fits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])
                    ]))
                # Setting the data of logical field via the
                # fits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header['EXTNAME'] = 'OI_VIS2'
                hdu.header['OI_REVN'] = revision, 'Revision number of the table definition'
                hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
                if key[0]: hdu.header['ARRNAME'] = key[0], 'Identifies corresponding OI_ARRAY'
                hdu.header['INSNAME'] = key[1], 'Identifies corresponding OI_WAVELENGTH table'
                hdulist.append(hdu)

        if self.t3.size:
            tables = {}
            # Check if any of the t3 tables are higher than revision 1; save
            # everything with highest revision used
            revision = 1
            for t3 in self.t3:
                if t3.revision > revision: revision = t3.revision
            for t3 in self.t3:
                nwave = t3.wavelength.eff_wave.size
                key = (arraymap.get(id(t3.array)), wavelengthmap.get(id(t3.wavelength)), corrmap.get(id(t3.corr)))
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          't3amp':[], 't3amperr':[], 't3phi':[], 't3phierr':[],
                                          'u1coord':[], 'v1coord':[], 'u2coord':[], 'v2coord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(targetmap[id(t3.target)])
                if t3.timeobs:
                    time = t3.timeobs - refdate
                    data['time'].append(time.days * 24.0 * 3600.0 + time.seconds)
                    mjd = (t3.timeobs - _mjdzero).days + (t3.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                    data['mjd'].append(mjd)
                else:
                    data['time'].append(None)
                    data['mjd'].append(None)
                data['int_time'].append(t3.int_time)
                if nwave == 1:
                    data['t3amp'].append(t3.t3amp[0])
                    data['t3amperr'].append(t3.t3amperr[0])
                    data['t3phi'].append(t3.t3phi[0])
                    data['t3phierr'].append(t3.t3phierr[0])
                    data['flag'].append(t3.flag[0])
                else:
                    data['t3amp'].append(t3.t3amp)
                    data['t3amperr'].append(t3.t3amperr)
                    data['t3phi'].append(t3.t3phi)
                    data['t3phierr'].append(t3.t3phierr)
                    data['flag'].append(t3.flag)
                data['u1coord'].append(t3.u1coord)
                data['v1coord'].append(t3.v1coord)
                data['u2coord'].append(t3.u2coord)
                data['v2coord'].append(t3.v2coord)
                if t3.station[0] and t3.station[1] and t3.station[2]:
                    data['sta_index'].append([stationmap[id(t3.station[0])], stationmap[id(t3.station[1])], stationmap[id(t3.station[2])]])
                else:
                    data['sta_index'].append([-1, -1, -1])
                if t3.corr:
                    raise NotImplementedError('Writing correlation information from OI_T3 tables is not yet implemented')
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
                    fits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    fits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    fits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                    fits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    fits.Column(name='T3AMP', format='%dD'%nwave, array=data['t3amp']),
                    fits.Column(name='T3AMPERR', format='%dD'%nwave, array=data['t3amperr']),
                    fits.Column(name='T3PHI', format='%dD'%nwave, unit='DEGREES', array=data['t3phi']),
                    fits.Column(name='T3PHIERR', format='%dD'%nwave, unit='DEGREES', array=data['t3phierr']),
                    fits.Column(name='U1COORD', format='1D', unit='METERS', array=data['u1coord']),
                    fits.Column(name='V1COORD', format='1D', unit='METERS', array=data['v1coord']),
                    fits.Column(name='U2COORD', format='1D', unit='METERS', array=data['u2coord']),
                    fits.Column(name='V2COORD', format='1D', unit='METERS', array=data['v2coord']),
                    fits.Column(name='STA_INDEX', format='3I', array=data['sta_index'], null=-1),
                    fits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])
                    )))
                # Setting the data of logical field via the
                # fits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header['EXTNAME'] = 'OI_T3'
                hdu.header['OI_REVN'] = revision, 'Revision number of the table definition'
                hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
                if key[0]: hdu.header['ARRNAME'] = key[0], 'Identifies corresponding OI_ARRAY'
                hdu.header['INSNAME'] = key[1], 'Identifies corresponding OI_WAVELENGTH table'
                hdulist.append(hdu)

        if self.flux.size:
            tables = {}
            revision = 1
            for flux in self.flux:
                nwave = flux.wavelength.eff_wave.size
                key = (arraymap.get(id(flux.array)), wavelengthmap.get(id(flux.wavelength)), corrmap.get(id(flux.corr)), flux.fov, flux.fovtype, flux.calibrated)
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'mjd':[], 'int_time':[],
                                          'fluxdata':[], 'fluxerr':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(targetmap[id(flux.target)])
                mjd = (flux.timeobs - _mjdzero).days + (flux.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                data['mjd'].append(mjd)
                data['int_time'].append(flux.int_time)
                if nwave == 1:
                    data['fluxdata'].append(flux.fluxdata[0])
                    data['fluxerr'].append(flux.fluxerr[0])
                    data['flag'].append(flux.flag[0])
                else:
                    data['fluxdata'].append(flux.fluxdata)
                    data['fluxerr'].append(flux.fluxerr)
                    data['flag'].append(flux.flag)
                if flux.station:
                    data['sta_index'].append(stationmap[id(flux.station)])
                else:
                    data['sta_index'].append(-1)
                if flux.corr:
                    raise NotImplementedError('Writing correlation information from OI_FLUX tables is not yet implemented')
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size
                
                cols = [fits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                       fits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                       fits.Column(name='INT_TIME', format='1D', array=data['int_time']),
                       fits.Column(name='FLUXDATA', unit=flux.fluxunit, format='%dD'%nwave, array=data['fluxdata']),
                       fits.Column(name='FLUXERR', unit=flux.fluxunit, format='%dD'%nwave, array=data['fluxerr'])]
                # Station should only be present for 'uncalibrated' spectra
                if not flux.calibrated:
                    cols += [fits.Column(name='STA_INDEX', format='1I', array=data['sta_index'], null=-1)]
                cols += [fits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])]
                hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))

                hdu.header['EXTNAME'] = 'OI_FLUX'
                hdu.header['OI_REVN'] = revision, 'Revision number of the table definition'
                hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
                hdu.header['INSNAME'] = key[1], 'Identifies corresponding OI_WAVELENGTH table'
                if key[0] and not flux.calibrated: hdu.header['ARRNAME'] = key[0], 'Identifies corresponding OI_ARRAY table'
                if key[2]: hdu.header['CORRNAME'] = key[2], 'Identifies corresponding OI_CORR table'
                if key[3] and flux.calibrated: hdu.header['FOV'] = key[3], 'Area over which flux is integrated (arcsec)'
                if key[4] and flux.calibrated: hdu.header['FOVTYPE'] = key[4], 'Model for FOV'
                if key[5]: hdu.header['CALSTAT'] = 'C', 'Calibration status'
                else: hdu.header['CALSTAT'] = 'U', 'Calibration status'
                hdulist.append(hdu)

        hdulist.writeto(filename, overwrite=overwrite)



def open(filename, quiet=False):
    """Open an OIFITS file."""

    newobj = oifits()
    targetmap = {}
    sta_indices = {}

    if not quiet:
        print("Opening %s"%filename)
    if type(filename) == fits.hdu.hdulist.HDUList:
        hdulist = filename
    else:
        hdulist = fits.open(filename)
    # Save the primary header
    newobj.header = hdulist[0].header.copy()

    # First get all the OI_TARGET, OI_WAVELENGTH, OI_ARRAY and OI_CORR tables
    for hdu in hdulist:
        header = hdu.header
        data = hdu.data
        # PyFITS 2.4 had a bug where strings in binary tables were padded with
        # spaces instead of nulls.  This was fixed in PyFITS 3.0.0, but many files
        # suffer from this problem, and the strings are ugly as a result.  Fix it.
        if type(hdu) == fits.hdu.table.BinTableHDU:
            for name in data.names:
                if data.dtype[name].type == np.string_:
                    data[name] = list(map(str.rstrip, data[name]))
        if hdu.name == 'OI_WAVELENGTH':
            revision = header['OI_REVN']
            insname = header['INSNAME']
            newobj.wavelength[insname] = OI_WAVELENGTH(data['EFF_WAVE'], data['EFF_BAND'], revision=revision)
        elif hdu.name == 'OI_TARGET':
            revision = header['OI_REVN']
            for row in data:
                target_id = row['TARGET_ID']
                if (revision >= 2) and ('CATEGORY' in data.names):
                    category = row['CATEGORY']
                else:
                    category = None
                target = OI_TARGET(target=row['TARGET'], raep0=row['RAEP0'], decep0=row['DECEP0'],
                                   equinox=row['EQUINOX'], ra_err=row['RA_ERR'], dec_err=row['DEC_ERR'],
                                   sysvel=row['SYSVEL'], veltyp=row['VELTYP'], veldef=row['VELDEF'],
                                   pmra=row['PMRA'], pmdec=row['PMDEC'], pmra_err=row['PMRA_ERR'],
                                   pmdec_err=row['PMDEC_ERR'], parallax=row['PARALLAX'],
                                   para_err=row['PARA_ERR'], spectyp=row['SPECTYP'], category=category, revision=revision)
                newobj.target = np.append(newobj.target, target)
                targetmap[target_id] = target
        elif hdu.name == 'OI_ARRAY':
            revision = header['OI_REVN']
            arrname = header['ARRNAME']
            frame = header['FRAME']
            arrxyz = np.array([header['ARRAYX'], header['ARRAYY'], header['ARRAYZ']])
            # Check if this file was written with an older verison (<0.5) of the module
            # and needs to have arrxyz positions fixed due to changing to ITRS
            if arrname == 'VLTI':
                for i, comment in enumerate(hdulist[0].header.get('COMMENT', '')):
                    if 'Written by OIFITS Python module' in str(comment):
                        if version.parse(comment.split()[-1]) < version.parse('0.5-dev'):
                            warnings.warn('Changing array center coordinates to ITRS', UserWarning)
                            oldheight = (np.sqrt((arrxyz**2).sum())-6378100.0)*u.m # lat/long are unchanged, but height is different
                            c = EarthLocation(*arrxyz*u.m)
                            c = EarthLocation(lat=c.lat, lon=c.lon, height=oldheight)
                            arrxyz = np.array([c.value[0], c.value[1], c.value[2]]) # c.value is numpy.void, which causes problems
                        break
            newobj.array[arrname] = OI_ARRAY(frame, arrxyz, stations=data, revision=revision)
            # Save the sta_index for each array, as we will need it
            # later to match measurements to stations
            sta_indices[arrname] = data['sta_index']
        elif hdu.name == 'OI_CORR':
            revision = header['OI_REVN']
            corrname = header['CORRNAME']
            newobj.corr[corrname] = OI_CORR(data['iindx'], data['jindx'], data['corr'], revision=revision)

    # Then get any science measurements
    for hdu in hdulist:
        header = hdu.header
        data = hdu.data
        if hdu.name in ('OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_FLUX', 'OI_INSPOL'):
            revision = header['OI_REVN']
            arrname = header.get('ARRNAME')
            array = newobj.array.get(arrname)
            corrname = header.get('CORRNAME')
            # INSPOL table has INSNAME in data, not in header
            if hdu.name != 'OI_INSPOL':
                wavelength = newobj.wavelength[header['INSNAME']]
            if 'T' in header['DATE-OBS']:
                warnings.warn('Warning: DATE-OBS contains a timestamp, which is contradictory to the OIFITS2 standard. Timestamp ignored.', UserWarning)
                date = header['DATE-OBS'].split('T')[0].split('-')
            else:
                date = header['DATE-OBS'].split('-')
        if hdu.name == 'OI_VIS':
            # OIFITS2 parameters which default to None for OIFITS1
            amptyp = phityp = amporder = phiorder = visrefmap = rvis = rviserr = ivis = iviserr = corr = None
            corrindx_visamp = corrindx_visphi = corrindx_rvis = corrindx_ivis = None
            ampunit = rvisunit = ivisunit = None
            if revision >= 2:
                amptyp = header.get('AMPTYP')
                phityp = header.get('PHITYP')
                amporder = header.get('AMPORDER')
                phiorder = header.get('PHIORDER')
                corr = newobj.corr.get(corrname)
                if amptyp == 'correlated flux':
                    ampunit = data.columns['VISAMP'].unit
                    # RVIS and IVIS may not be present
                    try:
                        rvisunit = data.columns['RVIS'].unit
                        ivisunit = data.columns['IVIS'].unit
                    except KeyError:
                        pass
            for row in data:
                timeobs = _mjdzero+datetime.timedelta(days=row['MJD'])
                int_time = row['INT_TIME']
                visamp = np.reshape(row['VISAMP'], -1)
                visamperr = np.reshape(row['VISAMPERR'], -1)
                visphi = np.reshape(row['VISPHI'], -1)
                visphierr = np.reshape(row['VISPHIERR'], -1)
                if 'CFLUX' in row.array.names: cflux = np.reshape(row['CFLUX'], -1)
                else: cflux = None
                if 'CFLUXERR' in row.array.names: cfluxerr = np.reshape(row['CFLUXERR'], -1)
                else: cfluxerr = None
                flag = np.reshape(row['FLAG'], -1)
                ucoord = row['UCOORD']
                vcoord = row['VCOORD']
                target = targetmap[row['TARGET_ID']]
                if array:
                    sta_index = row['STA_INDEX']
                    s1 = array.station[sta_indices[arrname] == sta_index[0]][0]
                    s2 = array.station[sta_indices[arrname] == sta_index[1]][0]
                    station = [s1, s2]
                else:
                    station = [None, None]
                # Optional OIFITS2 values
                if revision >= 2:
                    if 'VISREFMAP' in data.names:
                        visrefmap = row['VISREFMAP']
                    if 'RVIS' in data.names:
                        rvis = row['RVIS']
                    if 'RVISERR' in data.names:
                        rviserr = row['RVISERR']
                    if 'IVIS' in data.names:
                        ivis = row['IVIS']
                    if 'IVISERR' in data.names:
                        iviserr = row['IVISERR']
                    if 'CORRINDX_VISAMP' in data.names:
                        corrindx_visamp = row['CORRINDX_VISAMP']
                    if 'CORRINDX_VISPHI' in data.names:
                        corrindx_visphi = row['CORRINDX_VISPHI']
                    if 'CORRINDX_RVIS' in data.names:
                        corrindx_rvis = row['CORRINDX_RVIS']
                    if 'CORRINDX_IVIS' in data.names:
                        corrindx_ivis = row['CORRINDX_IVIS']
                newobj.vis = np.append(newobj.vis, OI_VIS(timeobs=timeobs, int_time=int_time, visamp=visamp,
                                                          visamperr=visamperr, visphi=visphi, visphierr=visphierr,
                                                          flag=flag, ucoord=ucoord, vcoord=vcoord, wavelength=wavelength, corr=corr,
                                                          target=target, array=array, station=station, cflux=cflux,
                                                          cfluxerr=cfluxerr, revision=revision,
                                                          corrindx_visamp=corrindx_visamp, corrindx_visphi=corrindx_visphi,
                                                          corrindx_rvis=corrindx_rvis, corrindx_ivis=corrindx_ivis,
                                                          amptyp=amptyp, phityp=phityp, amporder=amporder, phiorder=phiorder,
                                                          ampunit=ampunit, rvisunit=rvisunit, ivisunit=ivisunit,
                                                          visrefmap=visrefmap,
                                                          rvis=rvis, rviserr=rviserr, ivis=ivis, iviserr=iviserr))
        elif hdu.name == 'OI_VIS2':
            # OIFITS2 parameters which default to None for OIFITS1
            corr = corrindx_vis2data = None
            if revision >= 2:
                corr = newobj.corr.get(corrname)
            for row in data:
                timeobs = _mjdzero+datetime.timedelta(days=row['MJD'])
                int_time = row['INT_TIME']
                vis2data = np.reshape(row['VIS2DATA'], -1)
                vis2err = np.reshape(row['VIS2ERR'], -1)
                flag = np.reshape(row['FLAG'], -1)
                ucoord = row['UCOORD']
                vcoord = row['VCOORD']
                target = targetmap[row['TARGET_ID']]
                if array:
                    sta_index = row['STA_INDEX']
                    s1 = array.station[sta_indices[arrname] == sta_index[0]][0]
                    s2 = array.station[sta_indices[arrname] == sta_index[1]][0]
                    station = [s1, s2]
                else:
                    station = [None, None]
                # Optional OIFITS2 values
                if revision >= 2:
                    if 'CORRINDX_VIS2DATA' in data.names:
                        corrindx_vis2data = row['CORRINDX_VIS2DATA']
                newobj.vis2 = np.append(newobj.vis2, OI_VIS2(timeobs=timeobs, int_time=int_time, vis2data=vis2data,
                                                             vis2err=vis2err, flag=flag, ucoord=ucoord, vcoord=vcoord,
                                                             wavelength=wavelength, corr=corr, corrindx_vis2data=corrindx_vis2data,
                                                             target=target, array=array,
                                                             station=station, revision=revision))
        elif hdu.name == 'OI_T3':
            # OIFITS2 parameters which default to None for OIFITS1
            corr = corrindx_t3amp = corrindx_t3phi = None
            if revision >= 2:
                corr = newobj.corr.get(corrname)
            for row in data:
                timeobs = _mjdzero+datetime.timedelta(days=row['MJD'])
                int_time = row['INT_TIME']
                t3amp = np.reshape(row['T3AMP'], -1)
                t3amperr = np.reshape(row['T3AMPERR'], -1)
                t3phi = np.reshape(row['T3PHI'], -1)
                t3phierr = np.reshape(row['T3PHIERR'], -1)
                flag = np.reshape(row['FLAG'], -1)
                u1coord = row['U1COORD']
                v1coord = row['V1COORD']
                u2coord = row['U2COORD']
                v2coord = row['V2COORD']
                target = targetmap[row['TARGET_ID']]
                if array:
                    sta_index = row['STA_INDEX']
                    s1 = array.station[sta_indices[arrname] == sta_index[0]][0]
                    s2 = array.station[sta_indices[arrname] == sta_index[1]][0]
                    s3 = array.station[sta_indices[arrname] == sta_index[2]][0]
                    station = [s1, s2, s3]
                else:
                    station = [None, None, None]
                if revision >= 2:
                    if 'CORRINDX_T3AMP' in data.names:
                        corrindx_t3amp = row['CORRINDX_T3AMP']
                    if 'CORRINDX_T3PHI' in data.names:
                        corrindx_t3phi = row['CORRINDX_T3PHI']
                newobj.t3 = np.append(newobj.t3, OI_T3(timeobs=timeobs, int_time=int_time, t3amp=t3amp,
                                                       t3amperr=t3amperr, t3phi=t3phi, t3phierr=t3phierr,
                                                       flag=flag, u1coord=u1coord, v1coord=v1coord, u2coord=u2coord,
                                                       v2coord=v2coord, wavelength=wavelength, corr=corr,
                                                       corrindx_t3amp=corrindx_t3amp, corrindx_t3phi=corrindx_t3phi,
                                                       target=target, array=array, station=station, revision=revision))
        elif hdu.name == 'OI_FLUX':
            for row in data:
                timeobs = _mjdzero+datetime.timedelta(days=row['MJD'])
                int_time = row['INT_TIME']
                try:
                    fluxdata = np.reshape(row['FLUXDATA'], -1)
                except KeyError:
                    fluxdata = np.reshape(row['FLUX'], -1)
                    warnings.warn('Warning: This file does not conform to the OIFITS2 standard: OI_FLUX contains flux data in FLUX. Correcting to FLUXDATA.', UserWarning)
                fluxerr = np.reshape(row['FLUXERR'], -1)
                flag = np.reshape(row['FLAG'], -1)
                target = targetmap[row['TARGET_ID']]
                fov = header.get('FOV')
                fovtype = header.get('FOVTYPE')
                corr = newobj.corr.get(corrname)
                try:
                    fluxunit = data.columns['FLUXDATA'].unit
                except KeyError:
                    fluxunit = data.columns['FLUX'].unit
                    warnings.warn('Warning: This file does not conform to the OIFITS2 standard: OI_FLUX contains flux data in FLUX. Correcting to FLUXDATA.', UserWarning)
                fluxerrunit = data.columns['FLUXERR'].unit
                if header['CALSTAT'] == 'C':
                    calibrated = True
                else:
                    calibrated = False
                if array:
                    sta_index = row['STA_INDEX']
                    station = array.station[sta_indices[arrname] == sta_index][0]
                else:
                    station = None
                newobj.flux = np.append(newobj.flux,
                                        OI_FLUX(timeobs=timeobs, int_time=int_time,
                                        fluxdata=fluxdata, fluxerr=fluxerr, flag=flag,
                                        wavelength=wavelength, corr=corr, target=target,
                                        array=array, station=station, calibrated=calibrated,
                                        fov=fov, fovtype=fovtype, fluxunit=fluxunit, fluxerrunit=fluxerrunit,
                                        revision=revision))
        elif hdu.name == 'OI_INSPOL':
            for row in data:
                target = targetmap[row['TARGET_ID']]
                station = array.station[sta_indices[arrname] == row['STA_INDEX']][0]
                timestart = _mjdzero+datetime.timedelta(days=row['MJD_OBS'])
                timeend = _mjdzero+datetime.timedelta(days=row['MJD_END'])
                wavelength = newobj.wavelength[row['INSNAME']]
                newobj.inspol = np.append(newobj.inspol,
                                          OI_INSPOL(timestart, timeend, header['ORIENT'], header['MODEL'],
                                          row['JXX'], row['JYY'], row['JXY'], row['JYX'],
                                          wavelength, target, array, station, revision=revision))

    hdulist.close()
    if not quiet:
        newobj.info(recursive=False)

    return newobj
