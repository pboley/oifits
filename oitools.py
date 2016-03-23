import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import oifits
import warnings
import copy
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from matplotlib.patches import Rectangle
from numpy import sqrt, pi, arctan, sort

def match_wavelength(template, oifitsobj):

    oifitsobj = copy.deepcopy(oifitsobj)
    oifitsobj.wavelength = template.wavelength

    for tvis, vis in zip(template.vis, oifitsobj.vis):
        idx = np.argsort(vis.wavelength.eff_wave)
        vis._visamp = np.interp(tvis.wavelength.eff_wave, vis.wavelength.eff_wave[idx], vis._visamp[idx])
        vis._visamperr = np.interp(tvis.wavelength.eff_wave, vis.wavelength.eff_wave[idx], vis._visamperr[idx])
        vis._visphi = np.interp(tvis.wavelength.eff_wave, vis.wavelength.eff_wave[idx], vis._visphi[idx])
        vis._visphierr = np.interp(tvis.wavelength.eff_wave, vis.wavelength.eff_wave[idx], vis._visphierr[idx])
        if vis._cflux != None:
            vis._cflux = np.interp(tvis.wavelength.eff_wave, vis.wavelength.eff_wave[idx], vis._cflux[idx])
        if vis._cfluxerr != None:
            vis._visamperr = np.interp(tvis.wavelength.eff_wave, vis.wavelength.eff_wave[idx], vis._cfluxerr[idx])
        vis.flag = np.zeros_like(vis._visamp, dtype=bool)
        vis.wavelength = tvis.wavelength

    return oifitsobj

def average_vector(vector, npix):

    newvect = np.zeros(vector.size/npix, dtype=vector.dtype)
    endpoint = vector.size - vector.size % npix
    for i in range(npix):
        newvect += vector[i:endpoint:npix] / float(npix)

    return newvect

def average_wavelength(oifitsobj, npix):

    source = copy.deepcopy(oifitsobj)

    for wavelength in source.wavelength.values():
        wavelength.eff_wave = average_vector(wavelength.eff_wave, npix)
        wavelength.eff_band = average_vector(wavelength.eff_band, npix)

    for vis in source.vis:
        vis._visamp = average_vector(vis._visamp, npix)
        vis._visamperr = average_vector(vis._visamperr, npix)
        vis._visphi = average_vector(vis._visphi, npix)
        vis._visphierr = average_vector(vis._visphierr, npix)
        if vis._cflux != None:
            vis._cflux = average_vector(vis._cflux, npix)
        if vis._cfluxerr != None:
            vis._cfluxerr = average_vector(vis._cfluxerr, npix)
        vis.flag = average_vector(vis.flag, npix)

    for vis2 in source.vis2:
        vis2._vis2data = average_vector(vis2._vis2data, npix)
        vis2._vis2err = average_vector(vis2._vis2err, npix)
        vis2.flag = average_vector(vis2.flag, npix)

    for t3 in source.t3:
        t3._t3amp = average_vector(t3._t3amp, npix)
        t3._t3amperr = average_vector(t3._t3amperr, npix)
        t3._t3phi = average_vector(t3._t3phi, npix)
        t3._t3phierr = average_vector(t3._t3phierr, npix)
        t3.flag = average_vector(t3.flag, npix)

    return source


def plot_fullvis(oifitsobj):

    baseline = np.empty(0)
    visamp = np.empty(0)
    visamperr = np.empty(0)

    for vis in oifitsobj.vis:
        baseline = np.append(baseline, (vis.ucoord**2+vis.vcoord**2)/vis.wavelength.eff_wave[np.where(vis.flag == False)])
        visamp = np.append(visamp, vis.visamp[np.where(vis.flag == False)])
        visamperr = np.append(visamperr, vis.visamperr[np.where(vis.flag == False)])

    #plt.errorbar(baseline, visamp, visamperr, marker='dot', linestyle='none', color='black')
    plt.plot(baseline, visamp, 'k.')

def plot_full_visamp_vs_spatfreq(source, pmin=None, pmax=None, showerror=False):

    cmap = matplotlib.cm.rainbow

    fig = plt.figure()
    ax = fig.add_axes((0.125, 0.1, 0.9-0.125, 0.75))

    wave = source.vis[0].wavelength.eff_wave * 1e6

    posangles = np.empty(0)
    for vis in source.vis:
        posangle = np.arctan(vis.ucoord / vis.vcoord) * 180.0 / np.pi % 180.0
        posangles = np.append(posangles, posangle)

    if pmin == None: pmin = posangles.min()
    if pmax == None: pmax = posangles.max()
        

    for vis in source.vis:
        spatfreq = np.sqrt(vis.ucoord**2+vis.vcoord**2) / vis.wavelength.eff_wave / np.pi / 180.0 / 3600.0
        posangle = np.arctan(vis.ucoord / vis.vcoord) * 180.0 / np.pi
        posangle %= 180.0
        if showerror:
            ax.errorbar(spatfreq, vis.visamp, vis.visamperr, color=cmap((posangle - pmin) / (pmax - pmin)))
        else:
            ax.plot(spatfreq, vis.visamp, color=cmap((posangle - pmin) / (pmax - pmin)))

    ax.set_xlabel('Spatial frequency (fringe cycles/arcsec)')
    ax.set_ylabel('Visibility amplitude')
    ax.set_xlim(0,None)
    ax.set_ylim(0,None)

    position = ax.get_position().get_points()
    cbarax = fig.add_axes((position[0,0], 0.9, position[1,0]-position[0,0], 0.05), frameon=False)
    cbarax.imshow([np.linspace(0,1,100)], cmap=cmap, aspect='auto', extent=(pmin, pmax, 0, 1))

    plt.setp(cbarax.get_xticklabels(), visible=True)
    plt.setp(cbarax.get_yticklabels(), visible=False)
    plt.setp(cbarax.get_xticklines(), visible=True)
    plt.setp(cbarax.get_yticklines(), visible=False)
    cbarax.set_title('Position angle (deg)')

    fig.show()
    return posangles


def get_gaussian_width_from_visibility(baselines, visibilities):

    sigma = np.sqrt(-np.log(visibilities) / 2.0) / np.pi / baselines
    fwhm = np.sqrt(8.0 * np.log(2.0)) * sigma
    fwhm *= 180.0 / np.pi * 3600.0

    return fwhm

def plot_visamp_vs_spatfreq(source, waveidx=None, pmin=None, pmax=None, fig=None, marker='o'):

    cmap = matplotlib.cm.rainbow

    if waveidx == None:
        waveidx = np.arange(120,125)

    if np.isscalar(waveidx):
        waveidx = np.array([waveidx])

    wave = source.vis[0].wavelength.eff_wave * 1e6

    posangles = np.empty(0)
    for vis in source.vis:
        posangle = np.arctan(vis.ucoord / vis.vcoord) * 180.0 / np.pi % 180.0
        posangles = np.append(posangles, posangle)

    if fig == None:
        fig = plt.figure()
        ax = fig.add_axes((0.125, 0.1, 0.9-0.125, 0.75))
        if pmin == None: pmin = posangles.min()
        if pmax == None: pmax = posangles.max()
        position = ax.get_position().get_points()
        cbarax = fig.add_axes((position[0,0], 0.9, position[1,0]-position[0,0], 0.05), frameon=False)
        # These can be saved into the fig object for recall later...
        fig.pmin = pmin
        fig.pmax = pmax
    else:
        ax = fig.axes[0]
        pmin = fig.pmin
        pmax = fig.pmax
        cbarax = fig.axes[1]        

    for vis in source.vis:
        spatfreq = (np.sqrt(vis.ucoord**2+vis.vcoord**2) / vis.wavelength.eff_wave[waveidx] / np.pi / 180.0 / 3600.0).mean()
        visamp = vis.visamp[waveidx].mean()
        visamperr = np.sqrt((vis.visamperr[waveidx]**2).sum()/np.size(waveidx))
        posangle = np.arctan(vis.ucoord / vis.vcoord) * 180.0 / np.pi
        if posangle < 0: posangle += 180.0
        ax.errorbar([spatfreq], [visamp], [visamperr], color=cmap((posangle - pmin) / (pmax - pmin)), marker=marker)

    ax.set_xlabel('Spatial frequency (fringe cycles/arcsec)')
    ax.set_ylabel('Visibility amplitude')
    ax.text(0.98, 0.98, r'$\lambda=%.1f \mu$m'%(wave[waveidx].mean()), ha='right', va='top', transform=ax.transAxes)
    ax.set_xlim(0,None)
    ax.set_ylim(0,None)

    position = ax.get_position().get_points()
    cbarax = fig.add_axes((position[0,0], 0.9, position[1,0]-position[0,0], 0.05), frameon=False)
    cbarax.imshow([np.linspace(0,1,100)], cmap=cmap, aspect='auto', extent=(pmin, pmax, 0, 1))

    plt.setp(cbarax.get_xticklabels(), visible=True)
    plt.setp(cbarax.get_yticklabels(), visible=False)
    plt.setp(cbarax.get_xticklines(), visible=True)
    plt.setp(cbarax.get_yticklines(), visible=False)
    cbarax.set_title('Position angle (deg)')

    return fig

def plot_gaussian_widths(source, waveidx=100, legend=False):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for vis in source.vis:
        label = vis.station[0].sta_name + vis.station[1].sta_name
        baseline = np.sqrt(vis.ucoord**2 + vis.vcoord**2)
        fwhm = get_gaussian_width_from_visibility(baseline / vis.wavelength.eff_wave[waveidx], vis.visamp[waveidx])
        fwhm_upper = get_gaussian_width_from_visibility(baseline / vis.wavelength.eff_wave[waveidx], vis.visamp[waveidx]-vis.visamperr[waveidx])
        fwhm_lower = get_gaussian_width_from_visibility(baseline / vis.wavelength.eff_wave[waveidx], vis.visamp[waveidx]+vis.visamperr[waveidx])
        line = ax.plot([baseline], [fwhm], '.', label=label)
        ax.errorbar([baseline], [fwhm], [fwhm_upper-fwhm], color=line[0].get_color())
    if legend: plt.legend()
    ax.set_xlabel('Baseline (m)')
    ax.set_ylabel('FWHM (arcsec)')
    ax.set_title('%g $\\mu$m'%(source.vis[0].wavelength.eff_wave[waveidx]*1e6))
    fig.show()
    return fig

def get_ranges(data, value):

    ranges = []
    inrange = False
    i = 0

    while i < len(data):
        if data[i] == value and inrange == False:
            start = i
            inrange = True
        elif data[i] != value and inrange == True:
            ranges.append((start,i-1))
            inrange = False
        elif i == len(data) - 1 and inrange == True:
            ranges.append((start,i))
        i+=1

    return ranges
        

def plot_visamp_map(oifitsobj, width=0.1, height=0.1, colorcoding=None):

    fig = plt.figure(figsize=(10,8), facecolor='white')

    for vis in oifitsobj.vis:
        B = sqrt(vis.ucoord**2+vis.vcoord**2)
        pa = (arctan(vis.ucoord / vis.vcoord) * 180.0 / pi) % 180.0
        print B, pa
        ax = fig.add_axes((B/80.0-width/2.0, pa/180.0-height/2.0, width, height), axis_bgcolor=(0,0,0,0))
        if colorcoding:
            for key in sort(colorcoding.keys()):
                if B < key:
                    color = colorcoding[key]
                    break
        else: color='k'

        ax.errorbar(1e6*vis.wavelength.eff_wave, vis.visamp, vis.visamperr, fmt='k-', color=color)
        ax.set_xlim(8,13)
        ax.set_ylim(0,0.8)
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
        ax.xaxis.set_ticklabels((),)
        ax.yaxis.set_ticklabels((),)

    return fig

def plot_phases(oidata, uvplot=False, legend=False):

    xmin = None
    xmax = None
    ymin = 0.0
    ymax = 0.0
    baselinemax = None
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    if uvplot:
        fig2=plt.figure(figsize=(5,5))
        ax2=fig2.add_subplot(111)

    for vis in oidata.vis:
        ranges = get_ranges(vis.flag, True)
        for bounds in ranges:
            w1 = vis.wavelength.eff_wave[bounds[0]]*1e6
            w2 = vis.wavelength.eff_wave[bounds[1]]*1e6
            rect = matplotlib.patches.Rectangle((w1,-180), w2-w1, 360, color='lightblue')
            ax1.add_patch(rect)
    
    names = []

    colors = ('black', 'red', 'green', 'blue', 'purple', 'teal', 'orange', 'cyan', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'grey', 'darkgrey', 'lightgrey')
    colorid = 0

    for vis in oidata.vis:
        u = vis.ucoord
        v = vis.vcoord
        if (vis.station[0] and vis.station[1]):
            label = vis.station[0].sta_name + vis.station[1].sta_name
        else:
            label = 'unnamed'
        color = colors[colorid]
        colorid = np.mod(colorid + 1, len(colors))
        line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.visphi, vis.visphierr, label=label, color=color)

        ymin=np.amin(np.append(vis.visphi[np.where(vis.flag == False)], ymin))
        ymax=np.amax(np.append(vis.visphi[np.where(vis.flag == False)], ymax))
        xmin=np.amin(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmax))
        xmax=np.amax(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmin))
        if uvplot:
            ax2.plot([-u,u],[-v,v], '.', label=label, color=color)
        names.append(vis.target.target)
        baselinemax = np.amax([np.sqrt(u**2+v**2), baselinemax])
        print '%10s: %s, %5.2f m, %s'%(line[0].get_color(), label, np.sqrt(u**2+v**2), vis.timeobs)

    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(ymin,ymax)
    if uvplot:
        ax2.set_xlim(-1.1*baselinemax,1.1*baselinemax)
        ax2.set_ylim(-1.1*baselinemax,1.1*baselinemax)

    names = list(np.unique(names))
    title = names.pop()
    for name in names:
        title += ', %s'%(name)

    ax1.set_title(title)
    ax1.set_xlabel('Wavelength ($\mu$m)')
    ax1.set_ylabel('Differential phase')
    if uvplot:
        ax2.set_title(title)
        ax2.set_xlabel('u (m)')
        ax2.set_ylabel('v (m)')


    if legend and uvplot: ax2.legend(prop={'size':10},numpoints=1)

def plot_visibilities(oidata, uvplot=False, legend=False, ploterror=False):

    xmin = None
    xmax = None
    ymax = None
    baselinemax = None
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    if uvplot:
        fig2=plt.figure(figsize=(6,6))
        ax2=fig2.add_subplot(111)

    for vis in oidata.vis:
        ranges = get_ranges(vis.flag, True)
        for bounds in ranges:
            w1 = vis.wavelength.eff_wave[bounds[0]]*1e6
            w2 = vis.wavelength.eff_wave[bounds[1]]*1e6
            rect = matplotlib.patches.Rectangle((w1,0), w2-w1, 999, color='lightblue')
            ax1.add_patch(rect)
    
    names = []

    colors = ('black', 'red', 'green', 'blue', 'purple', 'teal', 'orange', 'cyan', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'grey', 'darkgrey', 'lightgrey', 'pink')
    colorid = 0
    output = ''

    for vis in oidata.vis:
        u = vis.ucoord
        v = vis.vcoord
        if (vis.station[0] and vis.station[1]):
            label = vis.station[0].sta_name + vis.station[1].sta_name
        else:
            label = 'unnamed'
        color = colors[colorid]
        colorid = np.mod(colorid + 1, len(colors))
        try:
            if ploterror:
                line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.visamp, vis.visamperr, label=label, color=color)
            else:
                line = ax1.plot(1e6*vis.wavelength.eff_wave, vis.visamp, label=label, color=color)
        except:
            if ploterror:
                line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.vis2data, vis.vis2err, label=label, color=color)
            else:
                line = ax1.plot(1e6*vis.wavelength.eff_wave, vis.vis2data, label=label, color=color)
        ymax=np.amax(np.append(vis.visamp.max(), ymax))
        xmin=np.amin(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmax))
        xmax=np.amax(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmin))
        if uvplot:
            ax2.plot([-u,u],[-v,v], '.', label=label, color=color)
        names.append(vis.target.target)
        baselinemax = np.amax([np.sqrt(u**2+v**2), baselinemax])
        output += '%10s: %s, %5.2f m, %s\n'%(line[0].get_color(), label, np.sqrt(u**2+v**2), vis.timeobs)

    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(0,ymax)
    if uvplot:
        ax2.set_xlim(-1.1*baselinemax,1.1*baselinemax)
        ax2.set_ylim(-1.1*baselinemax,1.1*baselinemax)

    names = list(np.unique(names))
    title = names.pop()
    for name in names:
        title += ', %s'%(name)

    ax1.set_title(title)
    ax1.set_xlabel('Wavelength ($\mu$m)')
    ax1.set_ylabel('Visibility')
    if uvplot:
        ax2.set_title(title)
        ax2.set_xlabel('u (m)')
        ax2.set_ylabel('v (m)')


    if legend and uvplot: ax2.legend(prop={'size':10},numpoints=1)
    print output

def plot_gaussian_widths_vs_wavelength(oidata, uvplot=False, legend=False, ploterror=False):

    xmin = None
    xmax = None
    ymax = None
    baselinemax = None
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    if uvplot:
        fig2=plt.figure(figsize=(6,6))
        ax2=fig2.add_subplot(111)

    for vis in oidata.vis:
        ranges = get_ranges(vis.flag, True)
        for bounds in ranges:
            w1 = vis.wavelength.eff_wave[bounds[0]]*1e6
            w2 = vis.wavelength.eff_wave[bounds[1]]*1e6
            rect = matplotlib.patches.Rectangle((w1,0), w2-w1, 999, color='lightblue')
            ax1.add_patch(rect)
    
    names = []

    colors = ('black', 'red', 'green', 'blue', 'purple', 'teal', 'orange', 'cyan', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'grey', 'darkgrey', 'lightgrey', 'pink')
    colorid = 0
    output = ''

    for vis in oidata.vis:
        u = vis.ucoord
        v = vis.vcoord
        if (vis.station[0] and vis.station[1]):
            label = vis.station[0].sta_name + vis.station[1].sta_name
        else:
            label = 'unnamed'
        color = colors[colorid]
        colorid = np.mod(colorid + 1, len(colors))
        widths = 1e3 * get_gaussian_width_from_visibility(sqrt(vis.ucoord**2 + vis.vcoord**2) / vis.wavelength.eff_wave, vis.visamp)
        line = ax1.plot(1e6*vis.wavelength.eff_wave, widths, label=label, color=color)
        ymax=np.amax(np.append(widths.max(), ymax))
        xmin=np.amin(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmax))
        xmax=np.amax(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmin))
        if uvplot:
            ax2.plot([-u,u],[-v,v], '.', label=label, color=color)
        names.append(vis.target.target)
        baselinemax = np.amax([np.sqrt(u**2+v**2), baselinemax])
        output += '%10s: %s, %5.2f m, %s\n'%(line[0].get_color(), label, np.sqrt(u**2+v**2), vis.timeobs)

    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(0,ymax)
    if uvplot:
        ax2.set_xlim(-1.1*baselinemax,1.1*baselinemax)
        ax2.set_ylim(-1.1*baselinemax,1.1*baselinemax)

    names = list(np.unique(names))
    title = names.pop()
    for name in names:
        title += ', %s'%(name)

    ax1.set_title(title)
    ax1.set_xlabel('Wavelength ($\mu$m)')
    ax1.set_ylabel('Gaussian width (mas)')
    if uvplot:
        ax2.set_title(title)
        ax2.set_xlabel('u (m)')
        ax2.set_ylabel('v (m)')


    if legend and uvplot: ax2.legend(prop={'size':10},numpoints=1)
    print output

def plot_cflux(oidata, uvplot=False, legend=False, ploterror=False):

    xmin = None
    xmax = None
    ymax = None
    baselinemax = None
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    if uvplot:
        fig2=plt.figure(figsize=(6,6))
        ax2=fig2.add_subplot(111)

    for vis in oidata.vis:
        ranges = get_ranges(vis.flag, True)
        for bounds in ranges:
            w1 = vis.wavelength.eff_wave[bounds[0]]*1e6
            w2 = vis.wavelength.eff_wave[bounds[1]]*1e6
            rect = matplotlib.patches.Rectangle((w1,0), w2-w1, 999, color='lightblue')
            ax1.add_patch(rect)
    
    names = []

    colors = ('black', 'red', 'green', 'blue', 'purple', 'teal', 'orange', 'cyan', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'grey', 'darkgrey', 'lightgrey', 'pink')
    colorid = 0
    output = ''

    for vis in oidata.vis:
        u = vis.ucoord
        v = vis.vcoord
        if (vis.station[0] and vis.station[1]):
            label = vis.station[0].sta_name + vis.station[1].sta_name
        else:
            label = 'unnamed'
        color = colors[colorid]
        colorid = np.mod(colorid + 1, len(colors))
        if ploterror:
            line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.cflux, vis.cfluxerr, label=label, color=color)
        else:
            line = ax1.plot(1e6*vis.wavelength.eff_wave, vis.cflux, label=label, color=color)
        ymax=np.amax(np.append(vis.cflux.max(), ymax))
        xmin=np.amin(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmax))
        xmax=np.amax(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmin))
        if uvplot:
            ax2.plot([-u,u],[-v,v], '.', label=label, color=color)
        names.append(vis.target.target)
        baselinemax = np.amax([np.sqrt(u**2+v**2), baselinemax])
        output += '%10s: %s, %5.2f m, %s\n'%(line[0].get_color(), label, np.sqrt(u**2+v**2), vis.timeobs)

    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(0,ymax)
    if uvplot:
        ax2.set_xlim(-1.1*baselinemax,1.1*baselinemax)
        ax2.set_ylim(-1.1*baselinemax,1.1*baselinemax)

    names = list(np.unique(names))
    title = names.pop()
    for name in names:
        title += ', %s'%(name)

    ax1.set_title(title)
    ax1.set_xlabel('Wavelength ($\mu$m)')
    ax1.set_ylabel('Correlated flux (Jy)')
    if uvplot:
        ax2.set_title(title)
        ax2.set_xlabel('u (m)')
        ax2.set_ylabel('v (m)')


    if legend and uvplot: ax2.legend(prop={'size':10},numpoints=1)
    print output

def plot_vis2(oidata, uvplot=False, legend=False, ploterror=False):

    xmin = None
    xmax = None
    ymax = None
    baselinemax = None
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    if uvplot:
        fig2=plt.figure(figsize=(6,6))
        ax2=fig2.add_subplot(111)

    for vis in oidata.vis2:
        ranges = get_ranges(vis.flag, True)
        for bounds in ranges:
            w1 = vis.wavelength.eff_wave[bounds[0]]*1e6
            w2 = vis.wavelength.eff_wave[bounds[1]]*1e6
            rect = matplotlib.patches.Rectangle((w1,0), w2-w1, 999, color='lightblue')
            ax1.add_patch(rect)
    
    names = []

    colors = ('black', 'red', 'green', 'blue', 'purple', 'teal', 'orange', 'cyan', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'grey', 'darkgrey', 'lightgrey', 'pink')
    colorid = 0

    for vis in oidata.vis2:
        u = vis.ucoord
        v = vis.vcoord
        if (vis.station[0] and vis.station[1]):
            label = vis.station[0].sta_name + vis.station[1].sta_name
        else:
            label = 'unnamed'
        color = colors[colorid]
        colorid = np.mod(colorid + 1, len(colors))
        if ploterror:
            line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.vis2data, vis.vis2err, label=label, color=color)
        else:
            print 1e6*vis.wavelength.eff_wave, vis.vis2data
            line = ax1.plot(1e6*vis.wavelength.eff_wave, vis.vis2data, label=label, color=color)
        ymax=np.amax(np.append(vis.vis2data[np.where(vis.flag == False)], ymax))
        xmin=np.amin(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmax))
        xmax=np.amax(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmin))
        if uvplot:
            ax2.plot([-u,u],[-v,v], '.', label=label, color=color)
        names.append(vis.target.target)
        baselinemax = np.amax([np.sqrt(u**2+v**2), baselinemax])
        print '%10s: %s, %5.2f m, %s'%(line[0].get_color(), label, np.sqrt(u**2+v**2), vis.timeobs)

    ax1.set_xlim(0.9*xmin,1.1*xmax)
    ax1.set_ylim(0,ymax)
    if uvplot:
        ax2.set_xlim(-1.1*baselinemax,1.1*baselinemax)
        ax2.set_ylim(-1.1*baselinemax,1.1*baselinemax)

    names = list(np.unique(names))
    title = names.pop()
    for name in names:
        title += ', %s'%(name)

    ax1.set_title(title)
    ax1.set_xlabel('Wavelength ($\mu$m)')
    ax1.set_ylabel('Visibility')
    if uvplot:
        ax2.set_title(title)
        ax2.set_xlabel('u (m)')
        ax2.set_ylabel('v (m)')


    if legend and uvplot: ax2.legend(prop={'size':10},numpoints=1)


def print_vis(vis):

    print "# %s %s %s"%(vis.target.target, vis.station[0].sta_name + vis.station[1].sta_name, vis.timeobs)
    print "# u = %6.2f v = %6.2f"%(vis.ucoord, vis.vcoord)
    print "# Wavelength (um)       "
    print "#        |V|"
    print "#                |V|_err"

    idx = list(np.where(vis.flag == False)[0])
    idx.reverse()
    
    for i in idx:
        print "%6.3f %8.4f %8.4f"%(1e6*vis.wavelength.eff_wave[i], vis.visamp[i], vis.visamperr[i])

def plot_array(array, visibilities=None):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    transOffset = offset_copy(ax.transData, fig=ax.figure, x = 0.0, y=-0.10, units='inches')

    center = array.arrxyz

    # Calculate a "east" unit vector
    east = np.array([center[1]*center[2],-center[0]*center[2],0])
    if center[2] > 0: east *= -1
    east /= np.sqrt(np.sum(np.square(east)))
    # Calculate a "north" unit vector
    north = np.cross(center, east)
    north /= np.sqrt(np.sum(np.square(north)))

    stations = array.station
    xlist = []
    ylist = []

    for station in stations:
        if 'U' in station.sta_name:
            color='green'
        else:
            color='blue'
        x = np.inner(station.staxyz, east)
        y = np.inner(station.staxyz, north)
        xlist.append(x)
        ylist.append(y)
        ax.plot([x], [y], 'o', color=color, markersize=1+station.diameter)
        plt.text(x, y, station.sta_name, transform=transOffset, horizontalalignment='center', verticalalignment='top', family='serif')

    if visibilities:
        for vis in visibilities:
            x = np.array([np.inner(vis.station[0].staxyz, east), np.inner(vis.station[1].staxyz, east)])
            y = np.array([np.inner(vis.station[0].staxyz, north), np.inner(vis.station[1].staxyz, north)])
            ax.plot(x, y, linestyle='-', marker='|', markersize=20, label=vis.station[0].sta_name + vis.station[1].sta_name)

    ax.plot([0], [0], 'r+')
    plt.text(0, 0, '$\\vec{O}$', transform=transOffset, horizontalalignment='center', verticalalignment='top')
    
    ax.annotate('N', xy=(0.05, 0.25), xytext=(0.05, 0.05), xycoords='axes fraction', textcoords='axes fraction', arrowprops={'width':2}, horizontalalignment='center', verticalalignment='bottom', family='serif', size='20')

    minx = np.min(xlist)
    miny = np.min(ylist)
    maxx = np.max(xlist)
    maxy = np.max(ylist)
    centerx = (maxx - minx) / 2.0 + minx
    centery = (maxy - miny) / 2.0 + miny
    
    width = 1.1*np.max([maxx - minx, maxy - miny])

    ax.set_xlim(centerx - width / 2.0, centerx + width / 2.0)
    ax.set_ylim(centery - width / 2.0, centery + width / 2.0)
    ax.relim()

    ax.set_xlabel('Relative position (m)')
    ax.set_ylabel('Relative position (m)')

def plot_uv_size(oidata, maxvis=0.3, waveidx=80):

    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(111)

    size = lambda visamp: visamp*15/maxvis+1

    for vis in oidata.vis:
        ax.plot([vis.ucoord], [vis.vcoord], marker='o', ms=size(vis.visamp[waveidx]))
        ax.plot([-vis.ucoord], [-vis.vcoord], marker='o', ms=size(vis.visamp[waveidx]))

    for visamp in np.linspace(0.0, maxvis, 10):
        ax.plot([-1000], [-1000], 'ko', ms=size(visamp), label='%.2g'%visamp)

    for radius in [10, 20, 30, 40, 50, 60]:
        x = np.linspace(-radius, radius, 100.0)
        y = np.sqrt(radius**2 - x**2)
        ax.plot(x, y, 'k:')
        ax.plot(x, -y, 'k:')

    ax.legend()

    ax.set_xlim(70,-70)
    ax.set_ylim(-70,70)
    ax.set_xlabel('u (m)')
    ax.set_ylabel('v (m)')
    ax.set_title(r'$\lambda = %g$ $\mu$m'%(1e6*oidata.vis[0].wavelength.eff_wave[waveidx]))

def station_distance(array, sta1name, sta2name):

    sta1 = array.get_station_by_name(sta1name)
    sta2 = array.get_station_by_name(sta2name)

    return sqrt(((sta1.staxyz-sta2.staxyz)**2).sum())
