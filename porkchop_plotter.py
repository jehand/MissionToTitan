# Porkchop Plotter - YouTube video
# AEON Grand Challenge
# Spring 2022

import matplotlib as mpl
from pykep.core import SEC2DAY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count, set_start_method
from csv import DictWriter, DictReader
from udps.chemical_mga import TitanChemicalMGAUDP
from udps.planetary_system2 import PlanetToSatellite
from trajectory_solver import TrajectorySolver, load_spice, spice_kernels
from itertools import repeat
from datetime import datetime as dt
from display_style import bcolors
from ast import literal_eval
import pykep as pk
from pykep.planet import jpl_lp

def norm(arr1, arr2):
    return np.sqrt(sum([(a-b)**2 for a,b in zip(arr1,arr2)]))

def interplanetary_porkchop(config):
    _config = {
        # Porkchop Plot Calculations
        'planet0'     : 'Earth',
        'planet1'     : 'Titan',
        'departure0'  : '2022-02-01',
        'departure1'  : '2040-12-31',
        'arrival0'    : '2023-01-01',
        'arrival1'    : '2050-12-31',
        'mu'          : 1,
        'step'        : 1 / SEC2DAY,
        'frame'       : 'ECLIPJ2000',
        'observer'    : 'SOLAR SYSTEM BARYCENTER',
        # Contour Plot Specs
        'cutoff_v'    : 20,
        'title'       : 'Earth to Titan Porkchop Plot',
        'fontsize'    : 15,
        'show' : True,
        'filename': None
    }

    for key in config.keys():
        _config[key] = config[key]

    cutoff_c3 = -config['cutoff_v'] ** 2

    # array of ephemeris time for departure and arrival times
    et_departures = np.arange(
        # utc2et converts calendar date to a spice ephemeris date
        spice.utc2et(_config['departure0']),
        spice.utc2et(_config['departure1']) + _config['step'],
        _config['step'])
    et_arrivals = np.arange(
        spice.utc2et(_config['arrival0']),
        spice.utc2et(_config[' arrival1']) + _config['step'],
        _config['step'])

    # number of days in each array and total combinations
    ds = len(et_departures)
    as_ = len(et_arrivals)
    total = ds*as_

    print('Departure days: %i.' % ds)
    print('Arrival days: %i.' % as_)
    print('Total Combinations: %i.' % total)

    # create a new empty array for C3, v_infinity, and tof
    C3_shorts = np.zeros((as_, ds))
    C3_longs = np.zeros((as_, ds))
    v_infinity_shorts = np.zeros((as_, ds))
    v_infinity_longs = np.zeros((as_, ds))
    tofs = np.zeros((as_, ds))

    # create arrays for indexing and meshgrid
    x = np.arange(ds)
    y = np.arange(as_)

    # for each combination...
    for na in y:
        for nd in x:
            # state of planet0 at departure - uses the calc_ephemeris function
            state_depart = st.calc_ephemeris(
                _config['planet0'],
                [et_departures[nd]],
                _config['frame'], _config['observer'])[0]

            # state of planet1 at arrival  - uses the calc_ephemeris function
            state_arrive = st.calc_ephemeris(
                _config['planet1'],
                [et_arrivals[na]],
                _config['frame'], _config['observer'])[0]

            # calculate flight time
            tof = et_arrivals[na] - et_departures[nd]

            try: # attempt to solve Lambert's problem with arrival, departure, and time of flight
                v_sc_depart_short, v_sc_arrive_short = lt.lamberts_universal_variables(
                    state_depart[:3], state_arrive[:3],
                    tof, tm = 1, mu = _config['mu'])
            except:
                v_sc_depart_short = np.array([1000, 1000, 1000])
                v_sc_arrive_short = np.array([1000, 1000, 1000])
            try:
                v_sc_depart_short, v_sc_arrive_short = lt.lamberts_universal_variables(
                    state_depart[:3], state_arrive[:3],
                    tof, tm = -1, mu = _config['mu'])
            except:
                v_sc_depart_long = np.array([1000, 1000, 1000])
                v_sc_arrive_long = np.array([1000, 1000, 1000])

            # calculate C3 values departing
            C3_short = norm(v_sc_depart_short,state_depart[3:]) ** 2      # "3:" means the velocity of the Earth
            C3_long = norm(v_sc_depart_long,state_depart[3:]) ** 2

            # check for unreasonable values (above the cutoff deltaV's)
            if C3_short > cutoff_c3: C3_short = cutoff_c3
            if C3_long > cutoff_c3: C3_long = cutoff_c3

            # calculate v infinity values arriving (subtract the difference and take the norm)
            v_inf_short = norm(v_sc_arrive_short,state_arrive[3:])
            v_inf_long = norm(v_sc_arrive_long,state_arrive[3:])

            # check for unreasonable values
            if v_inf_short > _config['cutoff_v']: v_inf_short = _config['cutoff_v']
            if v_inf_long > _config['cutoff_v']: v_inf_long = _config['cutoff_v']

            # add values to corresponding arrys (how you use the actual porkchop plot)
            C3_shorts[na, nd] = C3_short
            C3_longs[na, nd] = C3_long
            v_inf_shorts[na, nd] = v_inf_short
            v_inf_longs[na, nd] = v_inf_long
            tofs[na, nd] = tof

        # A print to see where it's at along the way that it's running
        print('%i / %i.' % (na, as_))

    # convert time of flight (tof) from seconds to days
    tofs /= (3600.0*24.0)

    # total deltaV
    dv_shorts = v_inf_shorts + np.sqrt(C3_shorts)
    dv_longs = v_inf_longs + np.sqrt(C3_longs)

    # create level arrays
    if _config['c3_levels'] is None:
        _config['c3_levels'] = np.arange(10, 50, 2)
    if _config['vinf_levels'] is None:
        _config['vinf_levels'] = nparange(0, 15, 1)
    if _config['tof_levels'] is None:
        _config['tof_levels'] = np.arange(100, 500, 20)
    if _config['dv_levels'] is None:
        _config['dv_levels'] = np.arange(3, 20, 0.5)

    lw = _config['lw']     # line width

    # Plotting the Porkchop Plot
    fig, ax = plt.subplots(figsize=_config['figsize'])
    c0 = ax.contour(C3_shorts,
        levels = _config['c3_levels'],  colors = 'm', linewidths = lw)     # m = magenta
    c1 = ax.contour(C3_longs,
        levels = _config['c#_levels'],  colors = 'm', linewidths = lw)
    c2 = ax.contour(v_inf_shorts,
        levels = _config['vinf_levels'], colors = 'deepbluesky', linewidths = lw)
    c3 = ax.contour(v_inf_longs,
        levels = _config['vinf_levels'],colors = 'deepskyblue', linewidths = lw)
    c4 = ax.contour(tofs,
        levels = _config['tof_levels'], colors = 'white', linewidths = lw * 0.6)

    # to label the contours (with an integer)
    plt.clabel(c0, fmt='%i')     # fmt = format, %i = module i which means give me an integer to label each contour
    plt.clabel(c1, fmt='%i')
    plt.clabel(c2, fmt='%i')
    plt.clabel(c3, fmt='%i')
    plt.clabel(c4, fmt='%i')
    # legend w/ code to make it look neat
    plt.plot([0], [0], 'm')
    plt.plot([0], [0], 'c')
    plt.plot([0], [0], 'w')
    # 3 things we want in the legend
    plt.legend(
        [r'C3 ($\dfrac{km^2}{s^2}$)',
        r'$V_{\infty}\; (\dfrac{km}{s})$',
        r'Time of Flight (days)'],
        # where we want the legend to be
        bbox_to_anchor = (1.005, 1.01),
        fontsize = 10)
    # label axes
    ax.set_title(_config['title'], fontsize = _config['fontsize'])
    ax.set_ylabel('Arrival (Days Past %s)' % _config['arrival0'], fontsize = _config['fontsize'])
    ax.set_xlabel('Departure (Days Past %s)' % _config['departure0'], fontsize = _config['fontsize'])

    if _config['show']:
        plt.show()

    if _config['filename'] is not None:

        plt.savefig['filename'], dpi = _config['dpi']
        print('Saved', _config['filename'])

    plt.close()

interplanetary_porkchop({
        # Porkchop Plot Calculations
        'planet0'     : 'Earth',
        'planet1'     : 'Titan',
        'departure0'  : '2022-02-01',
        'departure1'  : '2040-12-31',
        'arrival0'    : '2023-01-01',
        'arrival1'    : '2050-12-31',
        'mu'          : 1,
        'step'        : 1 / SEC2DAY,
        'frame'       : 'ECLIPJ2000',
        'observer'    : 'SOLAR SYSTEM BARYCENTER',
        # Contour Plot Specs
        'cutoff_v'    : 20,
        'title'       : 'Earth to Titan Porkchop Plot',
        'fontsize'    : 15,
    })

"""
    # Add in contours
    #...
    #delta V plot
    #...

    fig, ax = plt.subplot(figsize = _config[ 'figsize'])
    c0 = ax.contour(
        dv_shorts, levels = _config['dv_levels'],
        cmap = _config['dv_cmap'], linewidths = lw )
    c1 = ax.contour(
        dv_longs, levels = _config['dv_levels'],
        cmap = _config['dv_cmap'], linewidths = lw )
    c2 = ax.contour(tofs,
        levels = _config[ 'tof_levels' ], colors = 'c', linewidths = lw*0.6 )
    plt.clabel(c0, fmt = '%.1f')
    plt.clabel(c1, fmt = '%.1f')
    plt.clabel(c2, fmt = '%i')

    ax.set_title(_config['title_dv'], fontsize = _config['fontsize'])
    ax.set_ylabel('Arrival Days Past %s)' % _config['arrival0'], fontsize = _config['fontsize'])
    ax.set_xlabel('Departure (Days Past %s)' % _config['departure0'], fontsize = _config['fontsize'])

    if _config['show']:
        plt.show()
    if _config['filename_dv'] is not None:
        plt.savefig(_config['filename_dv'], dpi = _config['dpi'])
        print('Saved', _config['filename_dv'])

    plt.close()

"""
