import numpy as np
import oifits

# Modify (in place) OI_VIS tables in OIFITS1 file with CFLUX/CFLUXERR to OIFITS2
# with amptyp='correlated flux'
def cflux_to_oifits2(s):

    cflux = np.empty_like(s.vis)

    for i, vis in enumerate(s.vis):
        cflux[i] = oifits.OI_VIS(vis.timeobs, vis.int_time, vis._cflux, vis._cfluxerr,
                                 vis._visphi, vis._visphierr, vis.flag, vis.ucoord, vis.vcoord,
                                 vis.wavelength, vis.target, array=vis.array, station=vis.station, revision=2,
                                 amptyp='correlated flux')
        s.vis[i] = oifits.OI_VIS(vis.timeobs, vis.int_time, vis._visamp, vis._visamperr,
                                 vis._visphi, vis._visphierr, vis.flag, vis.ucoord, vis.vcoord,
                                 vis.wavelength, vis.target, array=vis.array, station=vis.station, revision=2)

    s.vis = np.append(s.vis, cflux)
