import oifits
from numpy.random import normal

def addnoise_oif(oif):

    for vis in oif.vis:
        vis._visamp[:] = normal(vis._visamp, vis._visamperr)
        vis._visphi[:] = normal(vis._visphi, vis._visphierr)
    
    for vis2 in oif.vis2:
        vis2._vis2data[:] = normal(vis2._vis2data, vis2._vis2err)
    
    for t3 in oif.t3:
        t3._t3amp[:] = normal(t3._t3amp, t3._t3amperr)
        t3._t3phi[:] = normal(t3._t3phi, t3._t3phierr)
    
    return oif