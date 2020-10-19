# Functions needed to run Excalibur
# Data munging in separate file
from os.path import basename
import numpy as np
from astropy.time import Time
from scipy import interpolate, optimize
from tqdm.auto import trange

import warnings
warnings.simplefilter('ignore', np.RankWarning)


###########################################################
# PCA Patching
###########################################################

def pcaPatch(x_values, mask, K=2, num_iters=50):
    """
    Iterative PCA patching, where bad values are replaced
    with denoised values.
    
    Parameters
    ----------
    x_values : 2D array
        List of values we want to denoise
    mask : 2D array
        Mask for x_values that is true for values that
        we would like to denoise
    K : int, optional (default: 2)
        Number of principal components used for denoising
    num_iters : int, optional (default: 50)
        Number of iterations to run iterative PCA patching
    
    Returns
    -------
    x_values : 2D ndarray
        x_values with bad values replaced by denoised values
    mean_x_values : 2D ndarray
        Mean of x_values over all exposures.
        Used as fiducial model of line positions
        PCA is done over deviations from this fiducial model
    denoised_xs : 2D ndarray
        Denoised x values from PCA reconstruction
    uu, ss, vv : ndarrays
        Arrays from single value decomposition.  Used to 
        reconstruct principal components and their corresponding
        coefficients
    """
    K = int(K)
    
    for i in range(num_iters):
        # There should be no more NaN values in x_values
        assert np.sum(np.isnan(x_values)) == 0
        # Redefine mean
        mean_x_values = np.mean(x_values,axis=0)
        
        # Run PCA
        uu,ss,vv = np.linalg.svd(x_values-mean_x_values, full_matrices=False)

        # Repatch bad data with K PCA reconstruction
        denoised_xs = mean_x_values + np.dot((uu*ss)[:,:K],vv[:K])
        x_values[mask] = denoised_xs[mask]
    
    return x_values, mean_x_values, denoised_xs, uu, ss, vv


def patchAndDenoise(x_values, orders, waves,
                    x_errors=None, times=None, file_list=None,
                    K=2, num_iters=50, running_window=9,
                    line_cutoff=0.5, file_cutoff=0.5,
                    outlier_cut=0, verbose=False):
    """
    - Vet for bad lines/exposures
    - Initial patch of bad data with running mean
    - Iterative patch with PCA for specified number of iterations
    - Optional second round of interative PCA patching to catch outliers
    
    
    Parameters
    ----------
    x_values, x_errors : 2D ndarray
        Array of line positions for all lines for each exposure and errors
    orders : 1D ndarray
        Array of orders for each line
    waves : 1D ndarray
        Array of wavelengths for each line
    times : 1D ndarray, optional
        Time stamps for each exposure
        Just written into the returned patch dictionary
        (not explicitely used for this code, but helps with evalWaveSol)
    K : int, optional (default: 2)
        Number of principal components used for denoising
    num_iters : int, optional (default: 50)
        Number of iterations to run iterative PCA patching
    running_window ; int, optional (default: 9)
        Window size of running mean used to initialize pixel values
        for lines missing measured pixel values
    line_cutoff, file_cutoff : float [0,1], optional (default: 0.5)
        Cutoff for bad lines or files, respectively.
        i.e. defaults cut lines that show up in less than 50% of exposure
        and files that contain less than 50% of all lines
    outlier_cut : float, optional (default: 0)
        Sigma cut used to identify outliers following first round of
        iterative PCA.
        Note: 0 means this process isn't done.
    
    Returns
    -------
    patch : dict
        Dictionary containing all the useful information from this process
        (among, very many uselss information!)
        Needed for evalWaveSol function
    """
    # Arrays that aren't needed, but are helpful to have in returned dictionary
    if times is None:
        times = np.zeros_like(file_list)
    if x_errors is None:
        x_errors = np.zeros_like(x_errors)
    if file_list is None:
        file_list = np.zeros_like(times)
    
    ### Vetting
    # Find where there is no line information
    x_values[np.nan_to_num(x_values) < 1] = np.nan
    
    # Mask out of order lines
    out_of_order = np.zeros_like(x_values,dtype=bool)
    for m in np.unique(orders):
        I = orders==m
        wave_sort = np.argsort(waves[I])
        for i, exp in enumerate(x_values):
            exp_sort = exp[I][wave_sort]
            exp_diff = np.diff(exp_sort)
            left_diff = np.insert(exp_diff<0,0,False)
            right_diff = np.append(exp_diff<0,False)
            exp_mask = np.logical_or(left_diff,right_diff)
            out_of_order[i,I] = exp_mask.copy()
    x_values[out_of_order] = np.nan
    if verbose:
        num_bad = np.sum(out_of_order)
        num_total = out_of_order.size
        print('{:.3}% of lines masked'.format(
             (num_bad)/num_total*100))
            
    # Get rid of bad lines
    good_lines = np.mean(np.isnan(x_values),axis=0) < line_cutoff
    # Trim everything
    orders = orders[good_lines]
    waves  = waves[good_lines]
    x_values = x_values[:,good_lines]
    x_errors = x_errors[:,good_lines]
    if verbose:
        num_good = np.sum(good_lines)
        num_total = good_lines.size
        print('{} of {} lines cut ({:.3}%)'.format(
            (num_total - num_good),num_total,
            (num_total - num_good)/num_total*100))
    
    # Get rid of bad files
    good_files = np.mean(np.isnan(x_values),axis=1) < file_cutoff
    # Trim everything
    x_values = x_values[good_files]
    x_errors = x_errors[good_files]
    file_names = file_list[good_files]
    file_times = times[good_files]
    if verbose:
        num_good = np.sum(good_files)
        num_total = good_files.size
        print('{} of {} files cut ({:.3}%)'.format(
            (num_total - num_good),num_total,
            (num_total - num_good)/num_total*100))
        print('Files that were cut:')
        print(file_list[~good_files])
    
    ### Patching
    # Initial patch of bad data with mean
    bad_mask = np.isnan(x_values) # mask to identify patched x_values
    if running_window > 0:
        half_size = int(running_window//2)
        counter = 6
        while np.sum(np.isnan(x_values)) > 0:
            for i in range(x_values.shape[0]):
                # Identify files in window
                file_range = [max((i-half_size,0)), min((i+half_size+1,x_values.shape[1]))]
                # Find mean of non-NaN values
                run_med = np.nanmean(x_values[file_range[0]:file_range[1],:],axis=0)
                # Patch NaN values with mean for center file
                x_values[i][bad_mask[i,:]] = run_med[bad_mask[i,:]]
            counter -= 1
            if counter < 0:
                print("Persistant NaNs with running mean.")
                print("Replacing remaining NaNs with global mean.")
                tot_mean = np.nanmean(x_values,axis=0)[None,...]*np.ones_like(x_values)
                x_values[np.isnan(x_values)] = tot_mean[np.isnan(x_values)]
                break
    else: # don't bother with running mean
        mean_values = np.nanmean(x_values,axis=0)
        mean_patch = np.array([mean_values for _ in range(x_values.shape[0])])
        x_values[bad_mask] = mean_patch[bad_mask]
    
    # Iterative PCA
    pca_results = pcaPatch(x_values, bad_mask, K=K, num_iters=num_iters)
    x_values, mean_x_values, denoised_xs, uu, ss, vv = pca_results
    
    # Mask line center outliers
    if outlier_cut > 0:
        x_resids  = x_values-denoised_xs
        out_mask  = abs(x_resids-np.mean(x_resids)) > (outlier_cut*np.nanstd(x_resids))
        if verbose:
            num_out = np.sum(out_mask)
            num_total = out_mask.size
            num_bad = np.sum(np.logical_and(out_mask,bad_mask))
            print('{:.3}% of lines marked as Outliers'.format(
                 (num_out)/num_total*100))
            print('{:.3}% of lines marked as Outliers that were PCA Patched'.format(
                 (num_bad)/num_total*100))
        pca_results = pcaPatch(x_values, np.logical_or(bad_mask,out_mask),
                               K=K, num_iters=num_iters)
        x_values, mean_x_values, denoised_xs, uu, ss, vv = pca_results
    
    # Load in all relevant information into dictionary
    patch_dict = {}
    patch_dict['K'] = K
    # Exposure Information
    patch_dict['files']  = file_names.copy()
    patch_dict['times']  = file_times.copy()
    min_date = Time(file_times.min(),format='mjd').isot.split('T')[0]
    yr, mn, dy = min_date.split('-')
    patch_dict['min_date'] = yr[2:]+mn+dy
    min_date = Time(file_times.max(),format='mjd').isot.split('T')[0]
    yr, mn, dy = min_date.split('-')
    patch_dict['max_date'] = yr[2:]+mn+dy
    # Line Information
    patch_dict['orders'] = orders.copy()
    patch_dict['waves']  = waves.copy()
    # Line Measurement Information
    patch_dict['x_values'] = x_values.copy()
    patch_dict['x_errors'] = x_errors.copy()
    patch_dict['denoised_xs'] = denoised_xs.copy()
    patch_dict['mean_xs']  = mean_x_values.copy()
    patch_dict['bad_mask'] = bad_mask.copy()
    # PCA Information
    patch_dict['u'] = uu.copy()
    patch_dict['s'] = ss.copy()
    patch_dict['v'] = vv.copy()
    patch_dict['ec'] = (uu*ss)[:,:K]
    # Outlier Information
    if outlier_cut > 0:
        patch_dict['out_mask'] = out_mask.copy()
    
    return patch_dict

# Functions for recovering the date of an exposure
def isot2date(isot_time):
    yr, mn, dy = isot_time.split('T')[0].split('-')
    return str(int(yr[2:]+mn+dy))

def mjds2dates(times):
    return np.array([isot2date(Time(t, format='mjd').isot) for t in times]).astype(str)

def files2dates(files):
    dates = []
    for file_name in files:
        date = basename(file_name).split('_')[-1].split('.')[0]
        dates.append(date)
    return np.array(dates).astype(str)

def interpPCA(new_interps, patch_dict, intp_deg=1, interp_key='times'):
    """
    Interpolate eigen coefficients with respect to chosen interp_key.
    
    Parameters
    ----------
    new_interps : 1D array or float
        New values for which we want principal component coefficients
    patch_dict : dictionary
        Result of patchAndDenoise function
    intp_deg : int, optional (default: 1)
        Degree of inteprolation
    interp_key : str, optional (default: 'times')
        Key in patch_dict of value we want to interpolate the principal
        component coeficients with respect to.
    
    Returns
    -------
    denoised_xs : 1D ndarray
        Pixel positions for calibration lines defined by order and 
        wavelength in the provided patch.
    """
    try:
        len(new_interps)
        unravel = False
    except TypeError:
        new_interps = [new_interps]
        unravel = True
    K  = patch_dict['K']
    vv = patch_dict['v']
    
    # Set up nightly code if needed
    if intp_deg=='poly':
        new_dates = mjds2dates(new_interps)
        cal_dates = files2dates(patch_dict['files'])
    
    # Interpolate eigen coefficients
    new_ecs = np.empty((len(new_interps),K),dtype=float)
    for i in range(K):
        if intp_deg=='poly':
            for date in np.unique(new_dates):
                new_date_mask = new_dates==date
                cal_date_mask = cal_dates==date
                if np.sum(cal_date_mask) < 5:
                    continue
                
                z = np.polyfit(patch_dict['times'][cal_date_mask][3:],
                               patch_dict['ec'][cal_date_mask][3:,i],3)
                new_ecs[new_date_mask,i] = np.poly1d(z)(new_interps[new_date_mask])
        
        elif intp_deg==0:
            # Find nearest time for each time
            # IS THERE A WAY TO DO THIS NOT ONE BY ONE???
            for i_idx, i in enumerate(new_interps):
                idx = np.abs(patch_dict[interp_key]-i).argmin()
                new_ecs[i_idx,i] = patch_dict['ec'][idx,i]
            
        elif intp_deg==1: # Default
            f = interpolate.interp1d(patch_dict[interp_key],patch_dict['ec'][:,i],kind='linear',
                                 bounds_error=False,fill_value=np.nan)
            new_ecs[:,i] = f(new_interps)
            
        elif intp_deg==3:
            f = interpolate.interp1d(patch_dict[interp_key],patch_dict['ec'][:,i],kind='cubic',
                                 bounds_error=False,fill_value=np.nan)
            new_ecs[:,i] = f(new_interps)
            
        else:
            tck = interpolate.splrep(patch_dict[interp_key],patch_dict['ec'][:,i],k=interp_deg)
            new_ecs[:,i] = interpolate.splev(new_interps,tck)
            
    # Construct x values for that period of time
    denoised_xs = np.dot(new_ecs,vv[:K]) + patch_dict['mean_xs']
    
    if unravel:
        return denoised_xs[0]
    else:
        return denoised_xs


###########################################################
# Interpolation Code
###########################################################

def interpWaveSol(newx, newm, x, m, data, interp_deg='pchip'):
    """
    Interpolate from known lines to construct a wavelength solution
    over new pixels and orders.  Proceeds order by order.
    
    Parameters
    ----------
    newx, newm : 1D arrays
        Line centers and matching orders for which to return wavelengths
    x, m, data : 1D arrays
        Line centers, matching orders, and known wavelengths for
        calibration lines
    interp_deg : str or int (1), optional (default: 'pchip')
        Specifies the type of interpolation
        - 'pchip', default, cubic spline with inforced monotonicity
        - 'inverse', fit wave vs. line position with a cubic spline
          (the idea was the constrain the cubic to be a function,
           but pchip does a better job and does it faster)
        - 1, linear interplation (I forget why this is still an option)
        - any other int, spline interpolation of specified degree
    
    Returns
    -------
    prediction : 1D ndarray
        Wavelengths for the input newx and newm
    """
    
    # Make all the searches match-able
    newm = np.array(newm,dtype=int)
    m = np.array(m,dtype=int)
    
    # Initialize prediction array
    prediction = np.zeros_like(newx)
    prediction[:] = np.nan
    # Go order by order
    for r in np.unique(m):
        Inew = newm == r
        I = m==r
        if (np.sum(Inew)>0) and (np.sum(I)>0):
            # Sort lines in increasing wavelength
            wave_sort = np.argsort(data[I])
            ord_xs = x[I][wave_sort]
            ord_data = data[I][wave_sort]
            # Make sure the lines are in order
            assert np.all(np.diff(ord_xs) > 0.),print(r,ord_xs[1:][np.argmin(np.diff(ord_xs))])
        
            # Interpolate
            if interp_deg==1: # linear interpolation
                prediction[Inew] = np.interp(newx[Inew], ord_xs, ord_data,
                                             left=np.nan,right=np.nan,k=interp_deg)
            elif interp_deg == 'pchip': # PCHIP interpolator
                f = interpolate.PchipInterpolator(ord_xs,ord_data,extrapolate=False)
                prediction[Inew] = f(newx[Inew])
            elif interp_deg == 'inverse': # Interpolating wave vs. pixel
                f = interpolate.interp1d(ord_data, ord_xs, kind='cubic',
                                         bounds_error=False,fill_value=0)
                inv_f = lambda x, a: f(x)-a
                
                f0 = interpolate.UnivariateSpline(ord_xs,ord_data,ext=1)
                
                predict = np.zeros(np.sum(Inew),dtype=float)
                for i,pix in enumerate(newx[Inew]):
                    if (pix <= ord_xs.min()) or (pix >= ord_xs.max()): # No extrapolation
                        predict[i] = np.nan
                    else:
                        try:
                            x0 = f0(pix)
                            predict[i] = optimize.newton(inv_f,x0,args=(pix,))
                        except RuntimeError:
                            predict[i] = np.nan
                prediction[Inew] = predict
            else: # spline interpolation
                tck = interpolate.splrep(ord_xs, ord_data, k=interp_deg)
                predict = interpolate.splev(newx[Inew],tck,ext=1)
                predict[predict==0] = np.nan
                prediction[Inew] = predict
    return prediction