"""
code.kazr_tools
=========================
Common core processing code for working with KAZR files

.. autosummary::
    :toctree: generated/
    hello_world
"""

import pyart
import os
import numpy as np
from matplotlib import pyplot as plt
import netCDF4
from scipy import ndimage, signal, integrate, interpolate
import time
import copy

def mean_with_gatefilter(radar, field, gatefilter, reverse=False):
    field_data = radar.fields[field]['data']
    if reverse:
        masked_field_data = np.ma.masked_where(gatefilter.gate_included,
                field_data)
    else:
        masked_field_data = np.ma.masked_where(gatefilter.gate_excluded,
                field_data)
    return masked_field_data.mean(axis=1)

def get_texture(radar):
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    start_time = time.time()
    data = ndimage.filters.generic_filter(radar.fields['mean_doppler_velocity']['data'],
                                                pyart.util.interval_std,
                                                size = (4,4),
                                               extra_arguments = (-nyq, nyq))
    total_time = time.time() - start_time
    print(total_time)
    filtered_data = ndimage.filters.median_filter(data, size = (4,4))
    texture_field = pyart.config.get_metadata('mean_doppler_velocity')
    texture_field['data'] = filtered_data
    return texture_field

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def snr_toa(radar):
    toa = 12.5*1000.0
    snr = radar.fields['snr_copol']['data']
    ranges = radar.range['data']
    cloud_top_gate = find_nearest(ranges, toa)
    mean_snr_clear = snr[:, cloud_top_gate::].mean(axis=1)
    return mean_snr_clear

def mean_snr(radar):
    snr = radar.fields['snr_copol']['data']
    mean_total = snr.mean(axis=1)
    return mean_total

def describe_radar(radar):
    texture =  get_texture(radar)
    radar.add_field('velocity_texture', texture, replace_existing = True)
    sig_dec = pyart.correct.GateFilter(radar)
    sig_dec.exclude_all()
    sig_dec.include_below('velocity_texture', 1.5)
    sig_returns = mean_with_gatefilter(radar, 'snr_copol', sig_dec)
    bg_returns = mean_with_gatefilter(radar, 'snr_copol', sig_dec, reverse=True)
    gates_sig = np.ones(radar.fields['snr_copol']['data'].shape)
    gates_sig[sig_dec.gate_excluded] = 0.0
    ngates_sig = gates_sig.sum(axis=1)
    return ngates_sig, sig_returns, bg_returns



