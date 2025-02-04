import oifits
import numpy as np
import datetime
from numpy import pi, cos, sin
from astropy.coordinates import SkyCoord, HADec
from astropy.time import Time
import astropy.units as u
import astroplan
from itertools import combinations

# P115 baselines
# small     A0-B2-D0-C1
# medium    KO-G2-D0-J3
# large     A0-G1-J2-K0
# extended  A0-B5-J2-J6
# Configurations to use (should be station names in the array). Note that any
# number of telescopes/configurations can be specified.
configs = (('A0', 'B2', 'D0', 'C1'), # small
           ('K0', 'G2', 'D0', 'J3'), # medium
           ('A0', 'G1', 'J2', 'K0'), # large
           ('A0', 'B5', 'J2', 'J6')) # extended

ra, dec = 255.0, -40.0 # RA=17h, dec=-40deg
d0 = Time('2025-06-15') # About the optimal date for RA of 17h
nHA = 6 # Number of hour angle spacings per configuration
ndays = 4 # Number of days per configuration
O = astroplan.Observer.at_site('paranal')

wave = np.linspace(3, 4, 10) # µm
wlname = 'MATISSE_L' # INSNAME for wavelength table

# Oloc: observer location (EarthLocation, e.g. O.location)
# Tloc: target location (SkyCoord)
# s1/s2: OI_STATION objects
# obstime: datetime object
def get_uv_coords(Oloc, Tloc, s1, s2, obstime):

    ha = Tloc.transform_to(HADec(obstime=obstime, location=Oloc)).ha.rad
    center = np.array(list(O.location.value))
    ra = Tloc.ra.rad
    dec = Tloc.dec.rad
    latitude = O.location.lat.rad
    longitude = O.location.lon.rad

    # Calculate an "east" unit vector
    east = np.array([center[1]*center[2],-center[0]*center[2],0])
    if center[2] > 0: east *= -1
    east /= np.sqrt(np.sum(east**2))
    # Calculate a "north" unit vector
    north = np.cross(center, east)
    north /= np.sqrt(np.sum(north**2))

    up = center / np.sqrt(np.sum(center**2))

    # Eq. 3 of Segransan 2007
    B = np.array([np.inner(s2.staxyz - s1.staxyz, north),
                  np.inner(s2.staxyz - s1.staxyz, east),
                  np.inner(s2.staxyz - s1.staxyz, up)])

    # From Lawson's book
    u = B[1] * cos(ha) - B[0] * sin(latitude) * sin(ha)
    v = B[1] * sin(dec) * sin(ha) + B[0] * (sin(latitude) * sin(dec) * cos(ha) + cos(latitude) * cos(dec))

    return u, v

def make_empty_obs(obstime, stations):

    vis = np.empty(0)
    vis2 = np.empty(0)
    t3 = np.empty(0)
    ucoords = []
    vcoords = []

    c = SkyCoord(oif.target[0].raep0*u.deg, oif.target[0].decep0*u.deg)
    ha = c.transform_to(HADec(obstime=obstime, location=O.location)).ha.rad
    nwl = len(wl.eff_wave)

    for sta1, sta2 in combinations(stations, 2):
        ucoord, vcoord = get_uv_coords(O.location, c, sta1, sta2, obstime)
        ucoords.append(ucoord)
        vcoords.append(vcoord)
        vis = np.append(vis, oifits.OI_VIS(obstime.to_datetime(), 60, np.ones(nwl), 0.05*np.ones(nwl),
                                           np.ones(nwl), 3*np.ones(nwl), np.zeros(nwl, dtype=bool),
                                           ucoord, vcoord, wl, target, array=array, station=(sta1, sta2)))
        vis2 = np.append(vis2, oifits.OI_VIS2(obstime.to_datetime(), 60, np.ones(nwl), 0.05*np.ones(nwl), np.zeros(nwl, dtype=bool),
                                              ucoord, vcoord, wl, target, array=array, station=(sta1, sta2)))

    for sta1, sta2, sta3 in combinations(stations, 3):
        u1coord, v1coord = get_uv_coords(O.location, c, sta1, sta2, obstime)
        u2coord, v2coord = get_uv_coords(O.location, c, sta2, sta3, obstime)
        ucoords.append(u1coord)
        ucoords.append(u2coord)
        vcoords.append(v1coord)
        vcoords.append(v2coord)
        t3 = np.append(t3, oifits.OI_T3(obstime.to_datetime(), 60, np.ones(nwl), 0.1*np.ones(nwl),
                                        np.ones(nwl), 2*np.ones(nwl), np.zeros(nwl, dtype=bool),
                                        u1coord, v1coord, u2coord, v2coord, wl, target, array=array, station=(sta1, sta2, sta3)))
    
    bls = np.sqrt(np.array(ucoords)**2+np.array(vcoords)**2)
    maxres = np.min(wave*1e-6) / np.max(bls) * 180 / pi * 3.6e6
    minres = np.max(wave*1e-6) / np.min(bls) * 180 / pi * 3.6e6

    print('Resolution %.2g-%.2g mas'%(maxres, minres))

    return vis, vis2, t3

merged = oifits.oifits()

for i, config in enumerate(configs):
    oif = oifits.open('../VLTI-array.fits')
    array = oif.array['VLTI']
    oif.target = np.array([oifits.OI_TARGET('fake source', ra, dec)])
    target = oif.target[0]
    wl = oif.wavelength[wlname] = oifits.OI_WAVELENGTH(wave*1e-6)
    obsdate = d0 + i*ndays*u.d
    oif.header['OBS-DATE'] = str(obsdate.to_datetime().date())
    nightstart = O.twilight_evening_astronomical(obsdate)
    nightend = O.twilight_morning_astronomical(obsdate)
    stations = [array.get_station_by_name(x) for x in config]
    for j in range(nHA):
        obstime = nightstart + j*(nightend-nightstart)/nHA
        obstime.format = 'datetime'
        vis, vis2, t3 = make_empty_obs(obstime, stations)
        oif.vis = np.append(oif.vis, vis)
        oif.vis2 = np.append(oif.vis2, vis2)
        oif.t3 = np.append(oif.t3, t3)
    
    oif.save('epoch%02d.fits'%i, overwrite=True)
    merged += oif

merged.save('merged.fits', overwrite=True)
