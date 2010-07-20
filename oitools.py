import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from matplotlib.patches import Rectangle

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

def plot_visibilities(oidata, uvplot=False, legend=False):

    xmin = None
    xmax = None
    ymax = None
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
            rect = matplotlib.patches.Rectangle((w1,0), w2-w1, 999, color='lightblue')
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
        try:
            line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.visamp, vis.visamperr, label=label, color=color)
        except:
            line = ax1.errorbar(1e6*vis.wavelength.eff_wave, vis.vis2data, vis.vis2err, label=label, color=color)
        ymax=np.amax(np.append(vis.visamp[np.where(vis.flag == False)], ymax))
        xmin=np.amin(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmax))
        xmax=np.amax(np.append(1e6*vis.wavelength.eff_wave[np.where(vis.flag == False)], xmin))
        if uvplot:
            ax2.plot([-u,u],[-v,v], '.', label=label, color=color)
        names.append(vis.target.target)
        baselinemax = np.amax([np.sqrt(u**2+v**2), baselinemax])
        print '%10s: %s, %5.2f m, %s'%(line[0].get_color(), label, np.sqrt(u**2+v**2), vis.timeobs)

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

    for id in stations:
        if 'U' in stations[id].sta_name:
            color='green'
        else:
            color='blue'
        x = np.inner(stations[id].staxyz, east)
        y = np.inner(stations[id].staxyz, north)
        xlist.append(x)
        ylist.append(y)
        ax.plot([x], [y], 'o', color=color, markersize=1+stations[id].diameter)
        plt.text(x, y, stations[id].sta_name, transform=transOffset, horizontalalignment='center', verticalalignment='top', family='serif')

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
    ax.set_title(array.arrname)
