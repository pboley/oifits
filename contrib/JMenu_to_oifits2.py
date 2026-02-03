# An example of converting the OIFITS1 files with correlated flux of Menu et al.
# (2015, A&A, 581, A107) to standard OIFITS2 files. The original files from that
# publication can be found on the OiDB at
# https://oidb.jmmc.fr/collection.html?id=9e568851-345e-42dc-aaf7-6ad0873e3b1b

import numpy as np
import oifits
from astropy.io import fits
from glob import glob
import datetime

# Modify (in place) OI_VIS tables in OIFITS1 file with CFLUX/CFLUXERR to OIFITS2
# with amptyp='correlated flux'; also extract the calibrated total spectra
# (which are saved with u=v=0)
def cflux_to_oifits2(s):

    newflux = []
    newcflux = []

    for vis in s.vis:
        if (vis.ucoord**2 + vis.vcoord**2) == 0:
            newflux.append(oifits.OI_FLUX(vis.timeobs, vis.int_time, vis._cflux, vis._cfluxerr,
                vis.flag, vis.wavelength, vis.target, True, 'Jy', 'Jy'))
        else:
            newcflux.append(oifits.OI_VIS(vis.timeobs, vis.int_time, vis._cflux, vis._cfluxerr,
                vis._visphi, vis._visphierr, vis.flag, vis.ucoord, vis.vcoord,
                vis.wavelength, vis.target, array=vis.array, station=vis.station, revision=2,
                amptyp='correlated flux', ampunit='Jy'))

    s.flux = np.array(newflux)
    s.vis = np.array(newcflux)


fnlist = glob('*.fits')

for fn in fnlist:
    hdulist = fits.open(fn)
    # OI_VIS table is mostly useless
    hdulist.pop('OI_VIS')
    # Take PRISM data; VISAMP/VISAMPERR here are placeholders and not used
    hdulist['OI_CORR_PRISM'].header['EXTNAME'] = 'OI_VIS'
    hdulist['OI_VIS'].columns.add_col(fits.Column(name='VISAMP', format=hdulist['OI_VIS'].columns['CFLUX'].format))
    hdulist['OI_VIS'].columns.add_col(fits.Column(name='VISAMPERR', format=hdulist['OI_VIS'].columns['CFLUX'].format))
    oifprism = oifits.open(hdulist)
    cflux_to_oifits2(oifprism)
    # Repeat for GRISM data
    hdulist = fits.open(fn)
    hdulist.pop('OI_WAVELENGTH')
    hdulist['OI_WAVELENGTH_GRISM'].header['EXTNAME'] = 'OI_WAVELENGTH'
    hdulist.pop('OI_VIS')
    hdulist['OI_CORR_GRISM'].header['EXTNAME'] = 'OI_VIS'
    hdulist['OI_VIS'].columns.add_col(fits.Column(name='VISAMP', format=hdulist['OI_VIS'].columns['CFLUX'].format))
    hdulist['OI_VIS'].columns.add_col(fits.Column(name='VISAMPERR', format=hdulist['OI_VIS'].columns['CFLUX'].format))
    oifgrism = oifits.open(hdulist)
    cflux_to_oifits2(oifgrism)
    oif = oifprism + oifgrism
    # Make updates for OIFITS2
    for key, array in oif.array.items():
        array.revision = 2
        for station in array.station:
            station.fovtype = 'RADIUS'
            if 'U' in station.sta_name:
                station.fov = 0.52 / 2
            else:
                station.fov = 2.29 / 2
    for target in oif.target:
        target.revision = 2
        target.category = 'SCI'
        target.veltyp = 'TOPOCENT'
        target.sysvel = np.nan
        target.veldef = 'OPTICAL'
    for key, wavelength in oif.wavelength.items():
        wavelength.revision = 2
    oif.header['INSMODE'] = 'HIGH_SENS'
    oif.header['ORIGIN'] = 'MPIA'
    oif.header['REFERNC'] = '2015A&A...581A.107M'
    oif.header['OBSERVER'] = 'UNKNOWN'
    obstime = datetime.datetime.utcnow()
    for vis in oif.vis:
        if vis.timeobs < obstime:
            obstime = vis.timeobs
    oif.header['DATE-OBS'] = obstime.strftime(format='%FT%T'), 'Timestamp of earliest observation (UTC)'
    oif.save('menu_cflux2/%s'%fn, overwrite=True)
