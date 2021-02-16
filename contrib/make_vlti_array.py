import numpy as np
from numpy import sin, cos, pi, sqrt
import oifits

# Coordinates of array center
longitude=-70.40479659*pi/180.0
latitude=-24.62794830*pi/180.0
R=2635.0+6378100.0

arrxyz = np.array([R*cos(latitude)*cos(longitude),
                   R*cos(latitude)*sin(longitude),
                   R*sin(latitude)])

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
stations = []

for sta_name, E, N in data:
    staxyz = [N*northx+E*eastx,
              N*northy+E*easty,
              N*northz]
    if 'U' in sta_name:
        diameter = 8.2
    else:
        diameter = 1.8
    stations.append((sta_name, sta_name, -1, diameter, staxyz))

array = oifits.OI_ARRAY('GEOCENTRIC', arrxyz, stations)

oifitsobj = oifits.oifits()
oifitsobj.array['VLTI'] = array

oifitsobj.save('VLTI-array.fits')
