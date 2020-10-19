# EXPRES-specific functions from which Excalibur emerges
# Ha ha, get it?

import os
from os.path import basename, join, isfile
from glob import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.constants import c
from astropy.time import Time
from scipy.io import readsav
from tqdm.auto import tqdm

# LFC Constants
rep_rate   = 14e9
lfc_offset = 6.19e9
###########################################################
# Functions for Reading in Data
###########################################################

def readParams(file_name):
    """
    Given the file name of a check_point file,
    load in all relevant data into 1D vectors
    
    Returns vectors for line center in pixel (x),
    order (y), error in line center fit in pixels (e),
    and wavelength of line (w)
    """
    try:
        info = np.load(file_name,allow_pickle=True)[()]
    except FileNotFoundError:
        if file_name.split('/')[-2] == 'checkpoint':
            lfc_id_dir = '/expres/extracted/lfc_cal/lfc_id/'
            file_name = lfc_id_dir + os.path.basename(file_name)
            info = np.load(file_name,allow_pickle=True)[()]
        else:
            raise FileNotFoundError
    # Assemble information into "fit-able" form
    num_orders = len(info['params'])
    lines = [p[:,1] for p in info['params'] if p is not None]
    errs = [np.sqrt(cov[:,1,1]) for cov in info['cov'] if cov is not None]
    ordrs = [o for o in np.arange(86) if info['params'][o] is not None]
    waves = [w for w in info['wvln'] if w is not None]
    # I believe, but am not sure, that the wavelengths are multiplied by order
    # to separate them from when orders overlap at the edges
    waves = [wvln for order, wvln in zip(ordrs,waves)]
    ordrs = [np.ones_like(x) * m for m,x in zip(ordrs, lines)]

    x = np.concatenate(lines)
    y = np.concatenate(ordrs)
    e = np.concatenate(errs)
    w = np.concatenate(waves)
    # Note: default of pipeline includes ThAr lines, which we're not including here
    
    return (x,y,w,e)

def readThid(thid_file):
    # Extract information from a thid file
    # And then very _very_ painfully order values and remove duplicates
    thid = readsav(thid_file)['thid']
    pixl = np.array(thid.pixel[0])
    ordr = np.array(160-thid.order[0])
    wave = np.array(thid.wvac[0])
    errs = np.array(thid.pix_err[0])
    
    sorted_pixl = []
    sorted_ordr = []
    sorted_wave = []
    for r in range(86):
        # Identify lines in the given order
        ord_mask = ordr==r
        sort_ordr = ordr[ord_mask]
        
        # Sort by wavelength along order
        ord_sort = np.argsort(pixl[ord_mask])
        sort_pixl = pixl[ord_mask][ord_sort]
        sort_wave = wave[ord_mask][ord_sort]
        sort_errs = errs[ord_mask][ord_sort]
        
        # Remove duplicate pixel values
        sorted_ordr.append([sort_ordr[0]])
        sorted_pixl.append([sort_pixl[0]])
        sorted_wave.append([sort_wave[0]])
        sorted_errs.append([sort_errs[0]])
        duplo = np.logical_and(np.diff(sort_pixl)!=0, np.diff(sort_wave)!=0)
        
        # Append to array
        sorted_ordr.append(sort_ordr[1:][duplo])
        sorted_pixl.append(sort_pixl[1:][duplo])
        sorted_wave.append(sort_wave[1:][duplo])
        sorted_errs.append(sort_errs[1:][duplo])
    return np.concatenate(sorted_pixl), np.concatenate(sorted_ordr), np.concatenate(sorted_wave), np.concatenate(sorted_errs)

def readFile(file_name):
    """
    Pick appropriate reading function based on file name
    Normalize output (thid doesn't have errors)
    """
    if file_name.endswith('thid'):
        x, m, w, e = readThid(file_name, pix_err=np.nan)
    else:
        x, m, w, e = readParams(file_name)
    return x, (160-m).astype(int), w, e


###########################################################
# Functions for Making Excalibur Accepted Data Structure
###########################################################

def mode2wave(modes):
    """
    Return true wavelengths given LFC modes
    """
    # True Wavelengths
    freq = modes * rep_rate + lfc_offset  # true wavelength
    waves = c.value / freq * 1e10 # magic
    return np.array(waves)

def wave2mode(waves):
    """
    Return LFC mode number given wavelengths
    """
    freq = c.value * 1e10 / waves
    modes = np.round((freq - lfc_offset) / rep_rate)
    return np.array(modes).astype(int)

def buildLineDB(file_list, flatten=True, verbose=False, use_orders=None,
                hand_mask=True):
    """ 
    Find all observed modes in specified file list
    
    Eventually we'll record mode number while lines are being fitted
        and this will be simplified like a whole bunch
    
    hand_mask: if True, remove lines we've determined are bad
        (stop gap measure)
    """
    # Load in all observed modes into a big dictionary
    name_dict = {}
    wave_dict = {}
    
    for file_name in tqdm(file_list, desc="Building Line DB"):
        try:
            x,m,w,e = readFile(file_name)
        except ValueError as err:
            if verbose:
                print(f'{os.path.basename(file_name)} threw error: {err}')
            continue
        for nord in np.unique(m):
            I = m==nord  # Mask for an order
            if nord not in name_dict:
                name_dict[nord] = []
                wave_dict[nord] = []
            # Get identifying names: "(order, wavelength string)"
            n = [(nord, wave_str(wave)) for wave in w[I]]
            name_dict[nord].extend(n)
            wave_dict[nord].extend(w[I])

    # Keep only the unique order-modes
    for nord in name_dict:
        name_dict[nord], unq_idx = np.unique(name_dict[nord], axis=0, return_index=True)
        wave_dict[nord] = np.asarray(wave_dict[nord])[unq_idx]

    # Combine all added names and waves into one long list
    name_keys, name_waves = [], []
    if use_orders is None:
        use_orders = list(name_dict.keys())
    for nord in use_orders:
        try:
            name_keys.append(name_dict[nord])
            name_waves.append(wave_dict[nord])
        except KeyError:
            continue
    name_keys = np.concatenate(name_keys)
    name_waves = np.concatenate(name_waves)
    
    if hand_mask:
        # Lines that are too close to be properly identified by thid.pro
        bad_orders = ['15','15','24','24','26','26','55','55','60','60','60']
        bad_wavels = [4231.05549,4231.05984, # 15
                      4482.07139,4482.08631, # 24
                      4545.13508,4545.14291, # 26
                      5841.11525,5841.10176, # 55
                      6116.61576,6116.61924,6140.35449] # 60
        for i in range(len(bad_orders)):
            bad_mask = np.logical_and(name_keys[:,0]==bad_orders[i],
                                      name_keys[:,1]==wave_str(bad_wavels[i]))
            if np.sum(bad_mask)>0:
                mask = np.ones(len(name_keys),dtype=bool)
                bad_indx = np.where(bad_mask)[0][0]
                mask[bad_indx] = False
                name_keys  = name_keys[mask]
                name_waves = name_waves[mask]
    if flatten:
        # Separate out names and orders
        orders, names = name_keys.T
        return orders.astype(int), names, name_waves
    else:
        return name_keys, name_waves

def getLineMeasures(file_list, orders, names, err_cut=0):
    """
    Find line center (in pixels) to match order/mode lines
    """
    # Load in x values to match order/mode lines
    x_values = np.full((len(file_list), len(orders)), np.nan)
    x_errors = np.full((len(file_list), len(orders)), np.nan)
    
    pd_keys = pd.DataFrame({'orders':orders.copy().astype(int),
                            'names':names.copy().astype(str)})
    for file_num, file_name in enumerate(tqdm(file_list, desc="Getting Line Measures")):
        # Load in line fit information
        try:
            x,m,w,e = readFile(file_name)
            if err_cut > 0:
                mask = e < err_cut
                x = x[mask]
                m = m[mask]
                w = w[mask]
                e = e[mask]
        except ValueError:
            continue
        
        # Identify which lines this exposure has
        for nord in np.unique(m):
            I = m==nord # Mask for an order
            # Get identifying names: "(nord, wavelength string)"
            n = [wave_str(wave) for wave in w[I]]
            xvl_dict = dict(zip(n,x[I]))
            err_dict = dict(zip(n,e[I]))
            ord_xval = np.array(pd_keys[pd_keys.orders==nord].names.map(xvl_dict))
            ord_errs = np.array(pd_keys[pd_keys.orders==nord].names.map(err_dict))
            x_values[file_num,pd_keys.orders==nord] = ord_xval
            x_errors[file_num,pd_keys.orders==nord] = ord_errs
                
    return x_values, x_errors


def wave_str(wave):
    return f"{wave:09.3f}"
