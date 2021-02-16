import oifits
import numpy as np
import datetime
from numpy import pi, cos, sin
from scipy.special import jv

def get_uv_coords(array, station1, station2, ra, dec, ha):

    # Convert hour angle, ra and dec to radians
    ha %= 24.0
    ha *= pi / 12.0
    center = array.arrxyz
    ra *= pi / 180.0
    dec *= pi / 180.0
    # Same for lat/long of array center
    latitude = vlti.latitude * pi / 180.0
    longitude = vlti.longitude * pi / 180.0

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

def uniform_disk(wave, diameter, baseline):

    spatfreq = baseline / wave
    diameter *= pi / 180.0 / 3.6e6

    return 2.0 * jv(1, pi * diameter * spatfreq) / pi / diameter / spatfreq

vlti = oifits.open('VLTI-array.fits').array['VLTI']
# Number of observations to generate
points = 15

# Values for calibrator star taken from van Boekel database
name = 'HD148478'
ra = 247.35192042 # deg
dec = -26.4320025 # deg
spectyp = 'M1.5Iab-b'
pmra = -2.822222179836697e-06 # deg/yr
pmdec = -6.4472219679090709e-06 # deg/yr
diameter = 34.84799957 # mas

savedata = np.load('midiwave.npz') # MIDI wavelength table

sample = oifits.oifits()
sample.wavelength['MIDI/PRISM'] = oifits.OI_WAVELENGTH(savedata.f.eff_wave, savedata.f.eff_band)
sample.array['VLTI'] = vlti
sample.target = np.append(sample.target, oifits.OI_TARGET(name, ra, dec, spectyp=spectyp, pmra=pmra,
                                                          pmdec=pmdec))

wavelength = sample.wavelength['MIDI/PRISM'].eff_wave
zeros = np.zeros_like(wavelength) # For errors, phase, etc. which I dont calculate
flag = np.zeros_like(wavelength, dtype=bool)
flag[(wavelength < 8e-6) | (wavelength > 13e-6)] = True

for k1, k2 in np.random.random_integers(len(vlti.station)-1, size=(points,2)):
    while k1 == k2:
        k1 = np.random.random_integers(len(vlti.station)-1)
    s1 = sample.array['VLTI'].station[k1]
    s2 = sample.array['VLTI'].station[k2]
    ha = np.random.random() * 10.0 - 5.0
    u, v = get_uv_coords(vlti, s1, s2, ra, dec, ha)
    visamp = np.abs(uniform_disk(wavelength, diameter, np.sqrt(u**2 + v**2)))
    sample.vis = np.append(sample.vis, oifits.OI_VIS(datetime.datetime.now(), 100, visamp, zeros, zeros, zeros, flag,
                                                     u, v, sample.wavelength['MIDI/PRISM'], sample.target[0],
                                                     array=sample.array['VLTI'], station=[s1,s2]))
    print('%s%s: (%g, %g)'%(s1.sta_name, s2.sta_name, u, v))

sample.save('%s.fits'%name)
