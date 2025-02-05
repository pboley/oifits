import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, sqrt
import oifits
from astropy.coordinates import EarthLocation
import astropy.units as u
from oitools import plot_array
from astropy.table import Table

# Coordinates of array center
longitude=-70.40479659*u.deg
latitude=-24.62794830*u.deg
R=2635.0*u.m

c = EarthLocation(lat=latitude, lon=longitude, height=R)
arrxyz = np.array([c.value[0], c.value[1], c.value[2]]) # c.value is numpy.void, which causes problems

arrayx, arrayy, arrayz = arrxyz

# Calculate an "east" unit vector
eastx=arrayy*arrayz
easty=-arrayx*arrayz
eastmag=sqrt(eastx*eastx+easty*easty)
eastx/=eastmag
easty/=eastmag

if arrayz > 0:
    eastx*=-1
    easty*=-1

# Calculate a "north" unit vector by crossing east and arrayxyz
northx=-easty*arrayz
northy=eastx*arrayz
northz=easty*arrayx-arrayy*eastx
northmag=sqrt(northx*northx+northy*northy+northz*northz)
northx/=northmag
northy/=northmag
northz/=northmag

data = np.genfromtxt('baseline_data.txt', usecols=(0,3,4), dtype=[('name', 'U2'), ('E', 'f8'), ('N', 'f8')])

# OI_ARRAY requires a named and iterable object similar to what's in OIFITS
# files.
t = Table()
t['STA_NAME'] = t['TEL_NAME'] = data['name']
t['DIAMETER'] = [8.2 if 'U' in x else 1.8 for x in data['name']]
t['STAXYZ'] = np.array([data['N']*northx+data['E']*eastx, data['N']*northy+data['E']*easty, data['N']*northz]).T

array = oifits.OI_ARRAY('GEOCENTRIC', arrxyz, t)

oifitsobj = oifits.oifits()
oifitsobj.array['VLTI'] = array

fig = plot_array(array)
fig.set_figheight(5)
fig.set_figwidth(5)
plt.tight_layout()
fig.savefig('VLTI-array.png')
oifitsobj.save('VLTI-array.fits')
