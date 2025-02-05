import numpy as np
import matplotlib.pyplot as plt
import oifits
import oimodeler as oim
import astropy.units as u
import astropy.coordinates as coords
from astropy.time import Time
from twobody import Barycenter, KeplerOrbit
from astropy.io import fits
from addnoise import addnoise_oif
import matplotlib.animation as animation
from os.path import join, splitext, isdir
from os import mkdir

# Either make an animation or generate synthetic data
doAnimation = False
cmap=plt.cm.hot

# oifn can be manually changed to generate synthetic data/plots from model
# based on file (saved to outdir with same name)
oifn = 'epoch03.fits'
outdir = 'output'

fov = 18 # mas
npix = 512
nt = 90 # number of time frames
extent = (fov/2, -fov/2, -fov/2, fov/2)

# Since distance is set to 1 kpc (below), 1 au = 1 ms
P = 20*u.day
a = 3*u.au
e = 0.2
i = 45*u.deg
omega = 15*u.deg

data = oim.oimData(oifn)
s = oifits.open(oifn, quiet=True)
origin = coords.ICRS(ra=s.target[0].raep0*u.deg, dec=s.target[0].decep0*u.deg, radial_velocity=0.0*u.km/u.s, distance=1000*u.pc)
orb = KeplerOrbit(P=P, e=e, a=a, i=i, omega=3*omega, Omega=3*omega, t0=Time('J2000.0'), barycenter=Barycenter(origin))

def getmodel(t):

    solution = orb.orbital_plane(t)
    x = solution.x.value
    y = solution.y.value
    f1 = 0.092
    f2 = 0.06
    x1 = x*f1/(f1+f2)
    y1 = y*f1/(f1+f2)
    x2 = -x*f2/(f1+f2)
    y2 = -y*f2/(f1+f2)
    pa = (np.arctan(x1/y1))
    if y1 < 0: pa += np.pi

    pt1 = oim.oimEGauss(x=x1, y=y1, f=f1, fwhm=0.35, elong=2.0, pa=35)
    pt2 = oim.oimEGauss(x=x2, y=y2, f=f2, fwhm=0.35, elong=1.5, pa=120)
    g = oim.oimGauss(fwhm=2, f=30)
    r  = oim.oimESKRing(x=0, y=0, f=1, din=10, dout=12, skw=1, skwPa=pa*180/np.pi, elong=1)
    c = oim.oimConvolutor(g, r)

    return oim.oimModel([pt1, pt2, c])


t0 = Time(s.header['OBS-DATE']) 

t = t0 + np.linspace(0, P, nt)

if doAnimation:
    images = [getmodel(x).getImage(npix, fov/npix, fromFT=True) for x in t]
    fig, ax = plt.subplots(figsize=(4,4))
    ims = []
    for i, image in enumerate(images):
        im = ax.imshow(image, extent=extent, cmap=cmap, animated=True)
        if i == 0:
            ax.imshow(image, extent=extent, cmap=cmap)
            ax.set_xlabel(r'$\Delta \alpha$ (mas)')
            ax.set_ylabel(r'$\Delta \delta$ (mas)')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=0)
    writer = animation.FFMpegWriter(fps=15, bitrate=720)
    ani.save('movie.gif', writer=writer)
else:
    m = getmodel(t0)
    sim = oim.oimSimulator(data=data, model=m)
    sim.compute(computeChi2=False, computeSimulatedData=True)
    ssim = oifits.open(sim.simulatedData.data[0], quiet=True)
    addnoise_oif(ssim)
    ssim.info()
    if not isdir(outdir): mkdir(outdir)
    ssim.save(join(outdir, oifn), overwrite=True)
    im = m.getImage(npix, fov/npix, fromFT=True)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(im, extent=extent, cmap=cmap)
    ax.set_xlabel(r'$\Delta \alpha$ (mas)')
    ax.set_ylabel(r'$\Delta \delta$ (mas)')
    #fig.savefig(join(outdir, splitext(oifn)[0] + '.png'))
    #np.save(join(outdir, splitext(oifn)[0] + '.npy'), im)
