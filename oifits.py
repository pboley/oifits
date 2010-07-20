"""
A module for reading/writing OIFITS files

This module is NOT related to the OIFITS Python module provided at
http://www.mrao.cam.ac.uk/research/OAS/oi_data/oifits.html
It is a (better) alternative.

To open an existing OIFITS file, use the oifits.open(filename)
function.  This will return an oifits object with the following
members (any of which can be None):

   array: a dictionary of interferometric arrays, as defined by the
   OI_ARRAY tables.  The dictionary key is the name of the array
   (ARRNAME).

   target: a dictionary of targets, as defined by the rows of the
   OI_TARGET table.  The dictionary key (an integer) corresponds to
   TARGET_ID.

   wavelength: a dictionary of wavelength tables (OI_WAVELENGTH).  The
   dictionary key is the name of the instrument/settings (INSNAME).

   vis, vis2 and t3: lists of objects containing all the measurement
   information.  Each list member corresponds to a row in an
   OI_VIS/OI_VIS2/OI_T3 table.

This module makes an ad-hoc, backwards-compatible change to the OIFITS
revision 1 standard originally described in Pauls et al., 2005, PASP,
117, 1255.  The OI_VIS and OI_VIS2 tables in OIFITS files produced by
this file contain two additional columns for the correlated flux,
CFLUX and CFLUXERR , which are arrays with a length corresponding to
the number of wavelength elements (just as VISAMP/VIS2DATA).

The main purpose of this module is to allow easy access to your OIFITS
data within Python, where you can then analyze it in any way you want.
It is not really intended for making changes to OIFITS data without
breaking the structure (e.g. the cross-references between tables), or
for creating OIFITS files from scratch.  However, the module also
provides a simple mechanism for combining multiple oifits objects,
achieved by using the '+' operator on two oifits objects: result = a +
b.  The result can then be written to a file using
result.save(filename).

For further information, contact Paul Boley (boley@mpia-hd.mpg.de).
   
"""

import numpy as np
import pyfits
import datetime
import copy

__author__ = "Paul Boley"
__email__ = "boley@mpia-hd.mpg.de"
__date__ ='20 July 2010'
__version__ = '0.2'
_mjdzero = datetime.datetime(1858, 11, 17)

matchtargetbyname = False
matchstationbyname = False
refdate = datetime.datetime(2000, 1, 1)

def _plurals(count):
    if (count > 1): return 's'
    return ''

def _array_eq(a, b):
    "Test whether all the elements of two arrays are equal."

    try:
        return not (a != b).any()
    except:
        return not (a != b)

class _angpoint(float):
    "Convenience object for angles"

    def __init__(self, angle):
        self.angle = angle

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

        hours = np.floor(angle)
        minutes = np.floor((angle - hours)*60.0)
        seconds = (angle - hours - minutes/60.0)*3600.0
        return "%02d:%02d:%05.2f"%(hours,minutes,seconds)

class OI_TARGET:

    def __init__(self, row):
        self.target = row['TARGET']
        self.raep0 = _angpoint(row['RAEP0'])
        self.decep0 = _angpoint(row['DECEP0'])
        self.equinox = row['EQUINOX']
        self.ra_err = row['RA_ERR']
        self.dec_err = row['DEC_ERR']
        self.sysvel = row['SYSVEL']
        self.veltyp = row['VELTYP']
        self.veldef = row['VELDEF']
        self.pmra = row['PMRA']
        self.pmdec = row['PMDEC']
        self.pmra_err = row['PMRA_ERR']
        self.pmdec_err = row['PMDEC_ERR']
        self.parallax = row['PARALLAX']
        self.para_err = row['PARA_ERR']
        self.spectyp = row['SPECTYP']

    def __eq__(self, other):

        if type(self) != type(other): return False
        
        return not (
            (self.target    != other.target)    or
            (self.raep0     != other.raep0)     or
            (self.decep0    != other.decep0)    or
            (self.equinox   != other.equinox)   or
            (self.ra_err    != other.ra_err)    or
            (self.dec_err   != other.dec_err)   or
            (self.sysvel    != other.sysvel)    or
            (self.veltyp    != other.veltyp)    or
            (self.veldef    != other.veldef)    or
            (self.pmra      != other.pmra)      or
            (self.pmdec     != other.pmdec)     or
            (self.pmra_err  != other.pmra_err)  or
            (self.pmdec_err != other.pmdec_err) or
            (self.parallax  != other.parallax)  or
            (self.para_err  != other.para_err)  or
            (self.spectyp   != other.spectyp))
            
    def __ne__(self, other):
        return not self.__eq__(other)

    def info(self):
        print "%s: %s %s (%g)"%(self.target, self.raep0.ashms(), self.decep0.asdms(), self.equinox)

    def get_id(self, parent):
        """Get the key of the given target in the parent oifits
        object.  Note that this key is an integer, and corresponds to
        the TARGET_ID value in the OIFITS specification."""

        for key in parent.target.keys():
            if parent.target[key] is self: return key


class OI_WAVELENGTH:

    def __init__(self, header, data):
        self.insname = header['INSNAME']
        self.eff_wave = data.field('EFF_WAVE')
        self.eff_band = data.field('EFF_BAND')

    def __eq__(self, other):

        if type(self) != type(other): return False
            
        return not (
            (not _array_eq(self.eff_wave, other.eff_wave)) or
            (not _array_eq(self.eff_band, other.eff_band)) or
            (self.insname != other.insname))

    def __ne__(self, other):
        return not self.__eq__(other)

    def info(self):
        print "'%s': %d wavelength%s (%.3g-%.3g um)"%(self.insname, len(self.eff_wave), _plurals(len(self.eff_wave)), 1e6*np.min(self.eff_wave),1e6*np.max(self.eff_wave))


class OI_VIS:

    def __init__(self, header, row, wavelength, target, array=None):
        date = header['DATE-OBS'].split('-')
        if len(date) == 3:
            self.timeobs = datetime.datetime(int(date[0]), int(date[1]), int(date[2])) + datetime.timedelta(seconds=np.around(row.field('TIME'), 2))
        else:
            self.timeobs = None
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = row.field('INT_TIME')
        # It seems single measurements aren't always saved as
        # arrays. This may be a PyFITS quirk but it doesn't work when
        # writing.
        try:
            len(row.field('VISAMP'))
            self.visamp = row.field('VISAMP')
            self.visamperr = row.field('VISAMPERR')
            self.visphi = row.field('VISPHI')
            self.visphierr = row.field('VISPHIERR')
            if 'CFLUX' in row.array.names:
                self.cflux = row.field('CFLUX')
            else:
                self.cflux = None
            if 'CFLUXERR' in row.array.names:
                self.cfluxerr = row.field('CFLUXERR')
            else:
                self.cfluxerr = None
            self.flag = row.field('FLAG')
        except:
            self.visamp = np.array([row.field('VISAMP')])
            self.visamperr = np.array([row.field('VISAMPERR')])
            self.visphi = np.array([row.field('VISPHI')])
            self.visphierr = np.array([row.field('VISPHIERR')])
            if 'CFLUX' in row.array.names:
                self.cflux = np.array([row.field('CFLUX')])
            else:
                self.cflux = None
            if 'CFLUXERR' in row.array.names:
                self.cfluxerr = np.array([row.field('CFLUXERR')])
            else:
                self.cfluxerr = None
            self.flag = np.array([row.field('FLAG')])
        self.ucoord = row.field('UCOORD')
        self.vcoord = row.field('VCOORD')
        if array:
            sta_index = row.field('STA_INDEX')
            self.station = [array.station[sta_index[0]], array.station[sta_index[1]]]
        else:
            self.station = [None, None]

    def __eq__(self, other):

        if type(self) != type(other): return False
        
        return not (
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.ucoord     != other.ucoord)     or
            (self.vcoord     != other.vcoord)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self.visamp, other.visamp)) or
            (not _array_eq(self.visamperr, other.visamperr)) or
            (not _array_eq(self.visphi, other.visphi)) or
            (not _array_eq(self.visphierr, other.visphierr)) or
            (not _array_eq(self.flag, other.flag)))
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def info(self):
        meanvis = np.mean(self.visamp[np.where(self.flag == False)])
        if self.station[0] and self.station[1]:
            baselinename = '(' + self.station[0].sta_name + self.station[1].sta_name + ')'
        else:
            baselinename = ''
        print "%s %s: %d point%s (%d masked), B = %4.1fm, <V> = %4.2g"%(self.target.target, baselinename, len(self.visamp), _plurals(len(self.visamp)), np.sum(self.flag), np.sqrt(self.ucoord**2 + self.vcoord**2), meanvis)

class OI_VIS2:

    def __init__(self, header, row, wavelength, target, array=None):
        date = header['DATE-OBS'].split('-')
        if len(date) == 3:
            self.timeobs = datetime.datetime(int(date[0]), int(date[1]), int(date[2])) + datetime.timedelta(seconds=np.around(row.field('TIME'), 2))
        else:
            self.timeobs = None
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = row.field('INT_TIME')
        # See comment in OI_VIS class for explanation of this
        try:
            len(row.field('VIS2DATA'))
            self.vis2data = row.field('VIS2DATA')
            self.vis2err = row.field('VIS2ERR')
            self.flag = row.field('FLAG')
            if 'CFLUX' in row.array.names:
                self.cflux = row.field('CFLUX')
            else:
                self.cflux = None
            if 'CFLUXERR' in row.array.names:
                self.cfluxerr = row.field('CFLUXERR')
            else:
                self.cfluxerr = None
        except:
            self.vis2data = np.array([row.field('VIS2DATA')])
            self.vis2err = np.array([row.field('VIS2ERR')])
            self.flag = np.array([row.field('FLAG')])
            if 'CFLUX' in row.array.names:
                self.cflux = np.array([row.field('CFLUX')])
            else:
                self.cflux = None
            if 'CFLUXERR' in row.array.names:
                self.cfluxerr = np.array([row.field('CFLUXERR')])
            else:
                self.cfluxerr = None
        self.ucoord = row.field('UCOORD')
        self.vcoord = row.field('VCOORD')
        if array:
            sta_index = row.field('STA_INDEX')
            self.station = [array.station[sta_index[0]], array.station[sta_index[1]]]
        else:
            self.station = [None, None]

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.ucoord     != other.ucoord)     or
            (self.vcoord     != other.vcoord)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self.vis2data, other.vis2data)) or
            (not _array_eq(self.vis2err, other.vis2err)) or
            (not _array_eq(self.flag, other.flag)))
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def info(self):
        meanvis = np.mean(self.vis2data[np.where(self.flag == False)])
        if self.station[0] and self.station[1]:
            baselinename = '(' + self.station[0].sta_name + self.station[1].sta_name + ')'
        else:
            baselinename = ''
        print "%s %s: %d point%s (%d masked), B = %4.1fm, <V> = %4.2g"%(self.target.target, baselinename, len(self.vis2data), _plurals(len(self.vis2data)), np.sum(self.flag), np.sqrt(self.ucoord**2 + self.vcoord**2), meanvis)


class OI_T3:

    def __init__(self, header, row, wavelength, target, array=None):
        date = header['DATE-OBS'].split('-')
        if len(date) == 3:
            self.timeobs = datetime.datetime(int(date[0]), int(date[1]), int(date[2])) + datetime.timedelta(seconds=np.around(row.field('TIME'), 2))
        else:
            self.timeobs = None
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = row.field('INT_TIME')
        # See comment in OI_VIS class for explanation.
        try:
            len(row.field('T3AMP'))
            self.t3amp = row.field('T3AMP')
            self.t3amperr = row.field('T3AMPERR')
            self.t3phi = row.field('T3PHI')
            self.t3phierr = row.field('T3PHIERR')
            self.flag = row.field('FLAG')
        except:
            self.t3amp = np.array([row.field('T3AMP')])
            self.t3amperr = np.array([row.field('T3AMPERR')])
            self.t3phi = np.array([row.field('T3PHI')])
            self.t3phierr = np.array([row.field('T3PHIERR')])
            self.flag = np.array([row.field('FLAG')])
        self.u1coord = row.field('U1COORD')
        self.v1coord = row.field('V1COORD')
        self.u2coord = row.field('U2COORD')
        self.v2coord = row.field('V2COORD')
        if array:
            sta_index = row.field('STA_INDEX')
            self.station = [array.station[sta_index[0]], array.station[sta_index[1]], array.station[sta_index[2]]]
        else:
            self.station = [None, None, None]

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
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
            (not _array_eq(self.t3amp, other.t3amp)) or
            (not _array_eq(self.t3amperr, other.t3amperr)) or
            (not _array_eq(self.t3phi, other.t3phi)) or
            (not _array_eq(self.t3phierr, other.t3phierr)) or
            (not _array_eq(self.flag, other.flag)))
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def info(self):
        meanvis = np.mean(self.t3amp[np.where(self.flag == False)])
        if self.station[0] and self.station[1] and self.station[2]:
            baselinename = '(' + self.station[0].sta_name + self.station[1].sta_name + self.station[2].sta_name + ')'
        else:
            baselinename = ''
        print "%s %s: %d point%s (%d masked), B = %4.1fm, %4.1fm, <V> = %4.2g"%(self.target.target, baselinename, len(self.t3amp), _plurals(len(self.t3amp)), np.sum(self.flag), np.sqrt(self.u1coord**2 + self.v1coord**2), np.sqrt(self.u2coord**2 + self.v2coord**2), meanvis)



class OI_STATION:
    """ This class corresponds to a single row (i.e. single
    station/telescope) of an OI_ARRAY table."""

    def __init__(self, sta_index, tel_name=None, sta_name=None, diameter=None, staxyz=[None, None, None]):
        self.sta_index = sta_index
        self.tel_name = tel_name
        self.sta_name = sta_name
        self.diameter = diameter
        self.staxyz = staxyz

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.tel_name != other.tel_name) or
            (self.sta_name != other.sta_name) or
            (self.diameter != other.diameter) or
            (not _array_eq(self.staxyz, other.staxyz)))

    def __ne__(self, other):
        return not self.__eq__(other)

class OI_ARRAY:
    """ Contains all the data for a single OI_ARRAY table. """

    def __init__(self, header, data):
        self.arrname = header['ARRNAME']
        self.frame = header['FRAME']
        self.arrxyz = np.array([header['ARRAYX'], header['ARRAYY'], header['ARRAYZ']])
        self.station = None
        for row in data:
            if self.station == None: self.station = {}
            sta_index = row['STA_INDEX']
            self.station[sta_index] = OI_STATION(sta_index, tel_name=row['TEL_NAME'], sta_name=row['STA_NAME'], diameter=row['DIAMETER'], staxyz=row['STAXYZ'])

    def __eq__(self, other):

        if type(self) != type(other): return False

        equal = not (
            (self.arrname != other.arrname) or
            (self.frame   != other.frame)   or
            (not _array_eq(self.arrxyz, other.arrxyz)))

        if not equal: return False
        
        # If arrname and position appear to be the same, check that
        # the stations (and ordering) are also the same
        if (self.station == None) and (other.station == None):
            return True
        if (self.station == None) or  (other.station == None):
            return False
        if len(self.station) != len(other.station):
            return False
        if self.station != other.station:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def info(self, verbose=0):
        """Print the array's center coordinates.  If verbosity >= 1,
        print information about each station."""
        radius = np.sqrt((self.arrxyz**2).sum())
        xylen = np.sqrt(self.arrxyz[0]**2+self.arrxyz[1]**2)
        latitude = _angpoint(np.arcsin(self.arrxyz[2]/radius)*180.0/np.pi)
        longitude = _angpoint(np.arcsin(self.arrxyz[1]/xylen)*180.0/np.pi)
        altitude = radius - 6378100.0
        print "'%s': %s %s %.4gm"%(self.arrname, latitude.asdms(), longitude.asdms(), altitude)
        if verbose >= 1:
            for key in self.station:
                print "   %.16s: %gm"%(self.station[key].sta_name + '/' + self.station[key].tel_name, self.station[key].diameter)
        
class oifits:
    
    def __init__(self):

        self.wavelength = None
        self.target = None
        self.array = None
        self.vis = None
        self.vis2 = None
        self.t3 = None

    def __add__(self, other):
        """Consistently combine two separate oifits objects.  Note
        that targets can be matched by name only (e.g. if coordinates
        differ) by setting oifits.matchtargetbyname to True.  The same
        goes for stations of the array (controlled by
        oifits.matchstationbyname)"""
        # Don't do anything of the two oifits objects are not CONSISTENT!
        if self.isconsistent() == False or other.isconsistent() == False:
            print 'oifits objects are not consistent, bailing.'
            return
        
        new = copy.deepcopy(self)
        if (other.wavelength):
            wavelengthmap = {}
            if new.wavelength == None: new.wavelength = {}
            for key in other.wavelength.keys():
                if key not in new.wavelength.keys():
                    new.wavelength[key] = copy.deepcopy(other.wavelength[key])
                elif new.wavelength[key] != other.wavelength[key]:
                    print 'Wavelength tables have the same key but differing contents'
                    return
                wavelengthmap[id(other.wavelength[key])]=new.wavelength[key]

        if other.target:
            targetmap = {}
            if new.target == None: new.target = {}
            for otarget in other.target.values():
                for ntarget in new.target.values():
                    if matchtargetbyname and ntarget.target == otarget.target:
                        targetmap[id(otarget)] = ntarget
                        break
                    elif ntarget == otarget:
                        targetmap[id(otarget)] = ntarget
                        break
                    elif ntarget.target == otarget.target:
                        print 'Found a target with a matching name, but some differences in the target specification.  Creating a new target.  Set oifits.matchtargetbyname to True to override this behavior.'
                # If 'id(otarget)' is not in targetmap, then this is a new
                # target and should be added to the dictionary of targets
                if id(otarget) not in targetmap.keys():
                    try:
                        newkey = new.target.keys()[-1]+1
                    except:
                        newkey = 1
                    new.target[newkey] = copy.deepcopy(otarget)
                    targetmap[id(otarget)] = new.target[newkey]

        if (other.array):
            if new.array == None: new.array = {}
            stationmap = {}
            for key in other.array.keys():
                if key not in new.array.keys():
                    new.array[key] = copy.deepcopy(other.array[key])
                # If arrays have the same name but seem to differ, try
                # to combine the two (by including the union of both
                # sets of stations)
                for othsta in other.array[key].station.values():
                    for newsta in new.array[key].station.values():
                        if newsta == othsta:
                            stationmap[id(othsta)] = newsta
                            break
                        elif matchstationbyname and newsta.sta_name == othsta.sta_name:
                            stationmap[id(othsta)] = newsta
                            break
                        elif newsta.sta_name == othsta.sta_name and matchstationbyname == False:
                            print 'Stations have matching names but conflicting data'
                            return
                    # If 'id(othsta)' is not in the stationmap
                    # dictionary, then this is a new station and
                    # should be added to the current array
                    if id(othsta) not in stationmap.keys():
                        try:
                            newkey = new.array[key].station.keys()[-1]+1
                        except:
                            newkey = 1
                        new.array[key].station[newkey] = copy.deepcopy(othsta)
                        # Make sure that staxyz of the new station is relative to the new array center
                        new.array[key].station[newkey].staxyz = othsta.staxyz - other.array[key].arrxyz + new.array[key].arrxyz
                        stationmap[id(othsta)] = new.array[key].station[newkey]
                    
        if (other.vis):
            if new.vis == None: new.vis = []
            for vis in other.vis:
                if vis not in new.vis:
                    new.vis.append(copy.copy(vis))
                    newvis = new.vis[-1]
                    # The wavelength, target, array and station objects
                    # should point to the appropriate objects inside
                    # the 'new' structure
                    newvis.wavelength = wavelengthmap[id(vis.wavelength)]
                    newvis.target = targetmap[id(vis.target)]
                    if (vis.array):
                        newvis.array = new.array[vis.array.arrname]
                        newvis.station = [None, None]
                        newvis.station[0] = stationmap[id(vis.station[0])]
                        newvis.station[1] = stationmap[id(vis.station[1])]

        if (other.vis2):
            if new.vis2 == None: new.vis2 = []
            for vis2 in other.vis2:
                if vis2 not in new.vis2:
                    new.vis2.append(copy.copy(vis2))
                    newvis2 = new.vis2[-1]
                    # The wavelength, target, array and station objects
                    # should point to the appropriate objects inside
                    # the 'new' structure
                    newvis2.wavelength = wavelengthmap[id(vis2.wavelength)]
                    newvis2.target = targetmap[id(vis2.target)]
                    if (vis2.array):
                        newvis2.array = new.array[vis2.array.arrname]
                        newvis2.station = [None, None]
                        newvis2.station[0] = stationmap[id(vis2.station[0])]
                        newvis2.station[1] = stationmap[id(vis2.station[1])]


        if (other.t3):
            if new.t3 == None: new.t3 = []
            for t3 in other.t3:
                if t3 not in new.t3:
                    new.t3.append(copy.copy(t3))
                    newt3 = new.t3[-1]
                    # The wavelength, target, array and station objects
                    # should point to the appropriate objects inside
                    # the 'new' structure
                    newt3.wavelength = wavelengthmap[id(t3.wavelength)]
                    newt3.target = targetmap[id(t3.target)]
                    if (t3.array):
                        newt3.array = new.array[t3.array.arrname]
                        newt3.station = [None, None, None]
                        newt3.station[0] = stationmap[id(t3.station[0])]
                        newt3.station[1] = stationmap[id(t3.station[1])]
                        newt3.station[2] = stationmap[id(t3.station[2])]

        
        return(new)
        

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.array      != other.array)      or
            (self.vis        != other.vis)        or
            (self.vis2       != other.vis2)       or
            (self.t3         != other.t3))

    def __ne__(self, other):
        return not self.__eq__(other)

    def isvalid(self):
        """Returns True of the oifits object is both consistent (as
        determined by isconsistent()) and conforms to the OIFITS
        standard (according to Pauls et al., 2005, PASP, 117, 1255)."""

        warnings = []
        errors = []
        if not self.isconsistent():
            errors.append('oifits object is not consistent')
        if self.target == None:
            errors.append('No OI_TARGET data')
        if self.wavelength == None:
            errors.append('No OI_WAVELENGTH data')
        else:
            for key in self.wavelength.keys():
                wavelength = self.wavelength[key]
                if key != wavelength.insname:
                    errors.append('Key (%s) and insname (%s) in wavelength object differ'%(key, wavelength.insname))
                if len(wavelength.eff_wave) != len(wavelength.eff_band):
                    errors.append("eff_wave and eff_band are of different lengths for wavelength table '%s'"%key)
        if (self.vis == None) and (self.vis2 == None) and (self.t3 == None):
            errors.append('Need to have atleast one measurement table (vis, vis2 or t3)')
        if self.vis:
            for vis in self.vis:
                nwave = len(vis.wavelength.eff_band)
                if (len(vis.visamp) != nwave) or (len(vis.visamperr) != nwave) or (len(vis.visphi) != nwave) or (len(vis.visphierr) != nwave) or (len(vis.flag) != nwave):
                    errors.append("Data size mismatch for visibility measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))
        if self.vis2:
            for vis2 in self.vis2:
                nwave = len(vis2.wavelength.eff_band)
                if (len(vis2.vis2data) != nwave) or (len(vis2.vis2err) != nwave) or (len(vis2.flag) != nwave):
                    errors.append("Data size mismatch for visibility^2 measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))                                  
        if self.t3:
            for t3 in self.t3:
                nwave = len(t3.wavelength.eff_band)
                if (len(t3.t3amp) != nwave) or (len(t3.t3amperr) != nwave) or (len(t3.t3phi) != nwave) or (len(t3.t3phierr) != nwave) or (len(t3.flag) != nwave):
                    errors.append("Data size mismatch for visibility measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))

        if warnings:
            print "*** %d warning%s:"%(len(warnings), _plurals(len(warnings)))
            for warning in warnings:
                print '  ' + warning
        if errors:
            print "*** %d ERROR%s:"%(len(errors), _plurals(len(errors)).upper())
            for error in errors:
                print '  ' + error

        return not (len(warnings) or len(errors))

    def isconsistent(self):
        """Returns True if the object is entirely self-contained,
        i.e. all cross-references to wavelength tables, arrays,
        stations etc. in the measurements refer to elements which are
        stored in the oifits object.  Note that an oifits object can
        be 'consistent' in this sense without being 'valid' as checked
        by isvalid()."""

        if (self.vis):
            for vis in self.vis:
                if vis.array and vis.array is not self.array[vis.array.arrname]:
                    print 'A visibility measurement (0x%x) refers to an array which is not inside the main oivis object.'%id(vis)
                    return False
                if vis.station[0] or vis.station[1]:
                    foundstation = [False, False]
                    for arrstation in vis.array.station.values():
                        if arrstation is vis.station[0]: foundstation[0] = True
                        if arrstation is vis.station[1]: foundstation[1] = True
                    if (vis.station[0] and foundstation[0] == False) or (vis.station[1] and foundstation[1] == False):
                        print foundstation
                        print 'A visibility measurement (0x%x) refers to a station which is not inside the main oivis object.'%id(vis)
                        return False
                if vis.wavelength == None:
                    print 'A visibility measurement (0x%x) has no wavelength object.'%id(vis)
                    return False
                if vis.wavelength is not self.wavelength[vis.wavelength.insname]:
                    print 'A visibility measurement (0x%x) refers to a wavelength table which is not inside the main oivis object.'%id(vis)
                    return False
                if vis.target == None:
                    print 'A visibility measurement (0x%x) has no target object.'%id(vis)
                    return False
                foundtarget = False
                for target in self.target.values():
                    if target is vis.target: foundtarget = True
                if foundtarget == False:
                    print 'A visibility measurement (0x%x) refers to a target which is not inside the main oivis table.'%id(vis)
                    return False

        if (self.vis2):
            for vis2 in self.vis2:
                if vis2.array and vis2.array is not self.array[vis2.array.arrname]:
                    print 'A visibility^2 measurement (0x%x) refers to an array which is not inside the main oivis2 object.'%id(vis2)
                    return False
                if vis2.station[0] or vis2.station[1]:
                    foundstation = [False, False]
                    for arrstation in vis2.array.station.values():
                        if arrstation is vis2.station[0]: foundstation[0] = True
                        if arrstation is vis2.station[1]: foundstation[1] = True
                    if (vis2.station[0] and foundstation[0] == False) or (vis2.station[1] and foundstation[1] == False):
                        print foundstation
                        print 'A visibility^2 measurement (0x%x) refers to a station which is not inside the main oivis2 object.'%id(vis2)
                        return False
                if vis2.wavelength == None:
                    print 'A visibility^2 measurement (0x%x) has no wavelength object.'%id(vis2)
                    return False
                if vis2.wavelength is not self.wavelength[vis2.wavelength.insname]:
                    print 'A visibility^2 measurement (0x%x) refers to a wavelength table which is not inside the main oivis2 object.'%id(vis2)
                    return False
                if vis2.target == None:
                    print 'A visibility^2 measurement (0x%x) has no target object.'%id(vis2)
                    return False
                foundtarget = False
                for target in self.target.values():
                    if target is vis2.target: foundtarget = True
                if foundtarget == False:
                    print 'A visibility^2 measurement (0x%x) refers to a target which is not inside the main oivis2 table.'%id(vis2)
                    return False

        if (self.t3):
            for t3 in self.t3:
                if t3.array and t3.array is not self.array[t3.array.arrname]:
                    print 'A closure phase measurement (0x%x) refers to an array which is not inside the main oit3 object.'%id(t3)
                    return False
                if t3.station[0] or t3.station[1]:
                    foundstation = [False, False]
                    for arrstation in t3.array.station.values():
                        if arrstation is t3.station[0]: foundstation[0] = True
                        if arrstation is t3.station[1]: foundstation[1] = True
                    if (t3.station[0] and foundstation[0] == False) or (t3.station[1] and foundstation[1] == False):
                        print foundstation
                        print 'A closure phase measurement (0x%x) refers to a station which is not inside the main oit3 object.'%id(t3)
                        return False
                if t3.wavelength == None:
                    print 'A closure phase measurement (0x%x) has no wavelength object.'%id(t3)
                    return False
                if t3.wavelength is not self.wavelength[t3.wavelength.insname]:
                    print 'A closure phase measurement (0x%x) refers to a wavelength table which is not inside the main oit3 object.'%id(t3)
                    return False
                if t3.target == None:
                    print 'A closure phase measurement (0x%x) has no target object.'%id(t3)
                    return False
                foundtarget = False
                for target in self.target.values():
                    if target is t3.target: foundtarget = True
                if foundtarget == False:
                    print 'A closure phase measurement (0x%x) refers to a target which is not inside the main oit3 table.'%id(t3)
                    return False
                    
        return True

    def info(self, recursive=True, verbose=0):
        """Print out a summary of the contents of the oifits object.
        Set recursive=True to obtain more specific information about
        each of the individual components, and verbose to an integer
        to increase the verbosity level."""

        if self.wavelength:
            wavelengths = 0
            if recursive:
                print "===================================================================="
                print "SUMMARY OF WAVELENGTH TABLES"
                print "===================================================================="
            for key in self.wavelength.keys():
                wavelengths += len(self.wavelength[key].eff_wave)
                if recursive: self.wavelength[key].info()
            print "%d wavelength table%s with %d wavelength%s in total"%(len(self.wavelength), _plurals(len(self.wavelength)), wavelengths, _plurals(wavelengths))
        if self.target:
            if recursive:
                print "===================================================================="
                print "SUMMARY OF TARGET TABLES"
                print "===================================================================="
                for key in self.target.keys():
                    self.target[key].info()
            print "%d target%s"%(len(self.target), _plurals(len(self.target)))
        if self.array:
            stations = 0
            if recursive:
                print "===================================================================="
                print "SUMMARY OF ARRAY TABLES"
                print "===================================================================="
            for key in self.array.keys():
                if recursive:
                    self.array[key].info(verbose=verbose)
                stations += len(self.array[key].station)
            print "%d array%s with %d station%s"%(len(self.array), _plurals(len(self.array)), stations, _plurals(stations))
        if self.vis:
            if recursive:
                print "===================================================================="
                print "SUMMARY OF VISIBILITY MEASUREMENTS"
                print "===================================================================="
                for vis in self.vis:
                    vis.info()
            print "%d visibility measurement%s"%(len(self.vis), _plurals(len(self.vis)))
        if self.vis2:
            if recursive:
                print "===================================================================="
                print "SUMMARY OF VISIBILITY^2 MEASUREMENTS"
                print "===================================================================="
                for vis2 in self.vis2:
                    vis2.info()
            print "%d visibility^2 measurement%s"%(len(self.vis2), _plurals(len(self.vis2)))
        if self.t3:
            if recursive:
                print "===================================================================="
                print "SUMMARY OF T3 MEASUREMENTS"
                print "===================================================================="
                for t3 in self.t3:
                    t3.info()
            print "%d closure phase measurement%s"%(len(self.t3), _plurals(len(self.t3)))

    def save(self, filename):
        """Write the contents of the oifits object to a file in OIFITS
        format."""

        if self.isconsistent() == False:
            print 'oifits object is not consistent, refusing to go further'
            return

        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        hdu.header.update('DATE', datetime.datetime.now().strftime(format='%F'), comment='Creation date')
        hdu.header.add_comment('Written by OIFITS Python module (Boley)')

        hdulist.append(hdu)
        if self.wavelength:
            for wavelength in self.wavelength.values():
                hdu = pyfits.new_table(pyfits.ColDefs((
                    pyfits.Column(name='EFF_WAVE', format='1E', unit='METERS', array=wavelength.eff_wave),
                    pyfits.Column(name='EFF_BAND', format='1E', unit='METERS', array=wavelength.eff_band)
                    )))
                hdu.header.update('EXTNAME', 'OI_WAVELENGTH')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('INSNAME', wavelength.insname, 'Name of detector, for cross-referencing')
                hdulist.append(hdu)

        if self.target:
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
            for key in self.target.keys():
                targ = self.target[key]
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

            hdu = pyfits.new_table(pyfits.ColDefs((
                pyfits.Column(name='TARGET_ID', format='1I', array=target_id),
                pyfits.Column(name='TARGET', format='16A', array=target),
                pyfits.Column(name='RAEP0', format='D1', unit='DEGREES', array=raep0),
                pyfits.Column(name='DECEP0', format='D1', unit='DEGREES', array=decep0),
                pyfits.Column(name='EQUINOX', format='E1', unit='YEARS', array=equinox),
                pyfits.Column(name='RA_ERR', format='D1', unit='DEGREES', array=ra_err),
                pyfits.Column(name='DEC_ERR', format='D1', unit='DEGREES', array=dec_err),
                pyfits.Column(name='SYSVEL', format='D1', unit='M/S', array=sysvel),
                pyfits.Column(name='VELTYP', format='A8', array=veltyp),
                pyfits.Column(name='VELDEF', format='A8', array=veldef),
                pyfits.Column(name='PMRA', format='D1', unit='DEG/YR', array=pmra),
                pyfits.Column(name='PMDEC', format='D1', unit='DEG/YR', array=pmdec),
                pyfits.Column(name='PMRA_ERR', format='D1', unit='DEG/YR', array=pmra_err),
                pyfits.Column(name='PMDEC_ERR', format='D1', unit='DEG/YR', array=pmdec_err),
                pyfits.Column(name='PARALLAX', format='E1', unit='DEGREES', array=parallax),
                pyfits.Column(name='PARA_ERR', format='E1', unit='DEGREES', array=para_err),
                pyfits.Column(name='SPECTYP', format='A16', array=spectyp)
                )))
            hdu.header.update('EXTNAME', 'OI_TARGET')
            hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
            hdulist.append(hdu)

        if self.array:
            for array in self.array.values():
                tel_name = []
                sta_name = []
                sta_index = []
                diameter = []
                staxyz = []
                if array.station:
                    for station in array.station.values():
                        tel_name.append(station.tel_name)
                        sta_name.append(station.sta_name)
                        sta_index.append(station.sta_index)
                        diameter.append(station.diameter)
                        staxyz.append(station.staxyz)
                    hdu = pyfits.new_table(pyfits.ColDefs((
                        pyfits.Column(name='TEL_NAME', format='16A', array=tel_name),
                        pyfits.Column(name='STA_NAME', format='16A', array=sta_name),
                        pyfits.Column(name='STA_INDEX', format='1I', array=sta_index),
                        pyfits.Column(name='DIAMETER', unit='METERS', format='1E', array=diameter),
                        pyfits.Column(name='STAXYZ', unit='METERS', format='3D', array=staxyz)
                        )))
                hdu.header.update('EXTNAME', 'OI_ARRAY')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('ARRNAME', array.arrname, comment='Array name, for cross-referencing')
                hdu.header.update('FRAME', array.frame, comment='Coordinate frame')
                hdu.header.update('ARRAYX', array.arrxyz[0], comment='Array center x coordinate (m)')
                hdu.header.update('ARRAYY', array.arrxyz[1], comment='Array center y coordinate (m)')
                hdu.header.update('ARRAYZ', array.arrxyz[2], comment='Array center z coordinate (m)')
                hdulist.append(hdu)
                        
        if self.vis:
            tables = {}
            for vis in self.vis:
                nwave = vis.wavelength.eff_wave.size
                if vis.array:
                    key = (vis.array.arrname, vis.wavelength.insname)
                else:
                    key = (None, vis.wavelength.insname)
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          'visamp':[], 'visamperr':[], 'visphi':[], 'visphierr':[],
                                          'cflux':[], 'cfluxerr':[], 'ucoord':[], 'vcoord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(vis.target.get_id(self))
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
                    if vis.cflux != None:
                        data['cflux'].append(vis.cflux[0])
                    else:
                        data['cflux'].append(None)
                    if vis.cfluxerr != None:
                        data['cfluxerr'].append(vis.cfluxerr[0])
                    else:
                        data['cfluxerr'].append(None)
                else:
                    data['visamp'].append(vis.visamp)
                    data['visamperr'].append(vis.visamperr)
                    data['visphi'].append(vis.visphi)
                    data['visphierr'].append(vis.visphierr)
                    data['flag'].append(vis.flag)
                    if vis.cflux != None:
                        data['cflux'].append(vis.cflux)
                    else:
                        cflux=np.empty(nwave)
                        cflux[:]=None
                        data['cflux'].append(cflux)
                    if vis.cfluxerr != None:
                        data['cfluxerr'].append(vis.cfluxerr)
                    else:
                        cfluxerr=np.empty(nwave)
                        cfluxerr[:]=None
                        data['cfluxerr'].append(cfluxerr)
                data['ucoord'].append(vis.ucoord)
                data['vcoord'].append(vis.vcoord)
                if vis.station[0] and vis.station[1]:
                    data['sta_index'].append([vis.station[0].sta_index,vis.station[1].sta_index])
                else:
                    data['sta_index'].append([-1, -1])
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = pyfits.new_table(pyfits.ColDefs([
                    pyfits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    pyfits.Column(name='MJD', unit='DAY', format='1D', array=data['mjd']),
                    pyfits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    pyfits.Column(name='VISAMP', format='%dD'%nwave, array=data['visamp']),
                    pyfits.Column(name='VISAMPERR', format='%dD'%nwave, array=data['visamperr']),
                    pyfits.Column(name='VISPHI', unit='DEGREES', format='%dD'%nwave, array=data['visphi']),
                    pyfits.Column(name='VISPHIERR', unit='DEGREES', format='%dD'%nwave, array=data['visphierr']),
                    pyfits.Column(name='CFLUX', format='%dD'%nwave, array=data['cflux']),
                    pyfits.Column(name='CFLUXERR', format='%dD'%nwave, array=data['cfluxerr']),
                    pyfits.Column(name='UCOORD', format='1D', unit='METERS', array=data['ucoord']),
                    pyfits.Column(name='VCOORD', format='1D', unit='METERS', array=data['vcoord']),
                    pyfits.Column(name='STA_INDEX', format='2I', array=data['sta_index'], null=-1),
                    pyfits.Column(name='FLAG', format='%dL'%nwave)
                    ]))

                # Setting the data of logical field via the
                # pyfits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header.update('EXTNAME', 'OI_VIS')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('DATE-OBS', refdate.strftime('%F'), comment='Zero-point for table (UTC)')
                if key[0]: hdu.header.update('ARRNAME', key[0], 'Identifies corresponding OI_ARRAY')
                hdu.header.update('INSNAME', key[1], 'Identifies corresponding OI_WAVELENGTH table')
                hdulist.append(hdu)

        if self.vis2:
            tables = {}
            for vis in self.vis2:
                nwave = vis.wavelength.eff_wave.size
                if vis.array:
                    key = (vis.array.arrname, vis.wavelength.insname)
                else:
                    key = (None, vis.wavelength.insname)
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          'vis2data':[], 'vis2err':[],
                                          'cflux':[], 'cfluxerr':[], 'ucoord':[], 'vcoord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(vis.target.get_id(self))
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
                    if vis.cflux != None:
                        data['cflux'].append(vis.cflux[0])
                    else:
                        data['cflux'].append(None)
                    if vis.cfluxerr != None:
                        data['cfluxerr'].append(vis.cfluxerr[0])
                    else:
                        data['cfluxerr'].append(None)
                else:
                    data['vis2data'].append(vis.vis2data)
                    data['vis2err'].append(vis.vis2err)
                    data['flag'].append(vis.flag)
                    if vis.cflux != None:
                        data['cflux'].append(vis.cflux)
                    else:
                        cflux=np.empty(nwave)
                        cflux[:]=None
                        data['cflux'].append(cflux)
                    if vis.cfluxerr != None:
                        data['cfluxerr'].append(vis.cfluxerr)
                    else:
                        cfluxerr=np.empty(nwave)
                        cfluxerr[:]=None
                        data['cfluxerr'].append(cfluxerr)
                data['ucoord'].append(vis.ucoord)
                data['vcoord'].append(vis.vcoord)
                if vis.station[0] and vis.station[1]:
                    data['sta_index'].append([vis.station[0].sta_index,vis.station[1].sta_index])
                else:
                    data['sta_index'].append([-1, -1])
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = pyfits.new_table(pyfits.ColDefs([
                    pyfits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    pyfits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                    pyfits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    pyfits.Column(name='VIS2DATA', format='%dD'%nwave, array=data['vis2data']),
                    pyfits.Column(name='VIS2ERR', format='%dD'%nwave, array=data['vis2err']),
                    pyfits.Column(name='CFLUX', format='%dD'%nwave, array=data['cflux']),
                    pyfits.Column(name='CFLUXERR', format='%dD'%nwave, array=data['cfluxerr']),
                    pyfits.Column(name='UCOORD', format='1D', unit='METERS', array=data['ucoord']),
                    pyfits.Column(name='VCOORD', format='1D', unit='METERS', array=data['vcoord']),
                    pyfits.Column(name='STA_INDEX', format='2I', array=data['sta_index'], null=-1),
                    pyfits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])
                    ]))
                # Setting the data of logical field via the
                # pyfits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header.update('EXTNAME', 'OI_VIS2')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('DATE-OBS', refdate.strftime('%F'), comment='Zero-point for table (UTC)')
                if key[0]: hdu.header.update('ARRNAME', key[0], 'Identifies corresponding OI_ARRAY')
                hdu.header.update('INSNAME', key[1], 'Identifies corresponding OI_WAVELENGTH table')
                hdulist.append(hdu)

        if self.t3:
            tables = {}
            for t3 in self.t3:
                nwave = t3.wavelength.eff_wave.size
                if t3.array:
                    key = (t3.array.arrname, t3.wavelength.insname)
                else:
                    key = (None, t3.wavelength.insname)
                if key in tables.keys():
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          't3amp':[], 't3amperr':[], 't3phi':[], 't3phierr':[],
                                          'u1coord':[], 'v1coord':[], 'u2coord':[], 'v2coord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(t3.target.get_id(self))
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
                    data['sta_index'].append([t3.station[0].sta_index, t3.station[1].sta_index, t3.station[2].sta_index])
                else:
                    data['sta_index'].append([-1, -1, -1])
            for key in tables.keys():
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = pyfits.new_table(pyfits.ColDefs((
                    pyfits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    pyfits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                    pyfits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    pyfits.Column(name='T3AMP', format='%dD'%nwave, array=data['t3amp']),
                    pyfits.Column(name='T3AMPERR', format='%dD'%nwave, array=data['t3amperr']),
                    pyfits.Column(name='T3PHI', format='%dD'%nwave, unit='DEGREES', array=data['t3phi']),
                    pyfits.Column(name='T3PHIERR', format='%dD'%nwave, unit='DEGREES', array=data['t3phierr']),
                    pyfits.Column(name='U1COORD', format='1D', unit='METERS', array=data['u1coord']),
                    pyfits.Column(name='V1COORD', format='1D', unit='METERS', array=data['v1coord']),
                    pyfits.Column(name='U2COORD', format='1D', unit='METERS', array=data['u2coord']),
                    pyfits.Column(name='V2COORD', format='1D', unit='METERS', array=data['v2coord']),
                    pyfits.Column(name='STA_INDEX', format='3I', array=data['sta_index'], null=-1),
                    pyfits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])
                    )))
                # Setting the data of logical field via the
                # pyfits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header.update('EXTNAME', 'OI_T3')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('DATE-OBS', refdate.strftime('%F'), 'Zero-point for table (UTC)')
                if key[0]: hdu.header.update('ARRNAME', key[0], 'Identifies corresponding OI_ARRAY')
                hdu.header.update('INSNAME', key[1], 'Identifies corresponding OI_WAVELENGTH table')
                hdulist.append(hdu)

        hdulist.writeto(filename, clobber=True)



def open(filename):
    """Open an OIFITS file."""
    
    newobj = oifits()
    
    print "Opening %s"%filename
    hdulist = pyfits.open(filename)
    # First get all the OI_TARGET, OI_WAVELENGTH and OI_ARRAY tables
    for hdu in hdulist:
        if hdu.name == 'OI_WAVELENGTH':
            if newobj.wavelength == None: newobj.wavelength = {}
            insname = hdu.header['INSNAME']
            newobj.wavelength[insname] = OI_WAVELENGTH(hdu.header, hdu.data)
        elif hdu.name == 'OI_TARGET':
            for row in hdu.data:
                if newobj.target == None: newobj.target = {}
                target_id = row['TARGET_ID']
                newobj.target[target_id] = OI_TARGET(row)
        elif hdu.name == 'OI_ARRAY':
            if newobj.array == None: newobj.array = {}
            arrname = hdu.header['ARRNAME']
            newobj.array[arrname] = OI_ARRAY(hdu.header, hdu.data)
    # Then get any science measurements
    for hdu in hdulist:
        if hdu.name == 'OI_VIS':
            arrname = hdu.header['ARRNAME']
            if arrname and newobj.array:
                array = newobj.array[arrname]
            else:
                array = None
            wavelength = newobj.wavelength[hdu.header['INSNAME']]
            for row in hdu.data:
                if newobj.vis == None: newobj.vis = []
                target = newobj.target[row.field('TARGET_ID')]
                newobj.vis.append(OI_VIS(hdu.header, row, wavelength, target, array=array))
        elif hdu.name == 'OI_VIS2':
            arrname = hdu.header['ARRNAME']
            if arrname and newobj.array:
                array = newobj.array[arrname]
            else:
                array = None
            wavelength = newobj.wavelength[hdu.header['INSNAME']]
            for row in hdu.data:
                if newobj.vis2 == None: newobj.vis2 = []
                target = newobj.target[row.field('TARGET_ID')]
                newobj.vis2.append(OI_VIS2(hdu.header, row, wavelength, target, array=array))
        elif hdu.name == 'OI_T3':
            arrname = hdu.header['ARRNAME']
            if arrname and newobj.array:
                array = newobj.array[arrname]
            else:
                array = None
            wavelength = newobj.wavelength[hdu.header['INSNAME']]
            for row in hdu.data:
                if newobj.t3 == None: newobj.t3 = []
                target = newobj.target[row.field('TARGET_ID')]
                newobj.t3.append(OI_T3(hdu.header, row, wavelength, target, array=array))
                    
    hdulist.close()
    newobj.info(recursive=False)

    return newobj
