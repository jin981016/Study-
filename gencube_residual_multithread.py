# gencube_residual_multithread
# 23.06.26.
# Minsu Kim @ Sejong Univ.

# DEPENDENCIES
# pip3 install numpy
# pip3 install scipy
# pip3 install astropy
# pip3 install tqdm
# pip3 install glob2
# pip3 install natsort

def task_multirun(z):

    import numpy as np
    from scipy.ndimage.filters import median_filter
    
    smoothed = median_filter(data_cube[z,:,:], size=size)
    np.save(path_temp+'/{}.npy'.format(z), smoothed)

        
def run_main(n_cores, path_cube, size_kernel, area_crop=[0,0,0,0]):

    # PARAMETERS
    # n_cores:      (int) number of threads to be used
    # path_cube:    (str) path to the cube
    # size_kernel:  (int, odd) kernel size
    # area_crop:    (list of ints) area to be used, [xi, xf, yi, yf]
    #                             if [0,0,0,0], full area will be used

    import os
    import shutil
    from astropy.io import fits
    import multiprocessing
    from tqdm import tqdm
    from natsort import natsorted
    import glob
    import numpy as np

    global data_cube, size, path_temp

    wdir = os.path.dirname(path_cube)
    path_temp = wdir+'/temp_smoothing'

    if(os.path.exists(path_temp)):
        shutil.rmtree(path_temp)

    os.mkdir(path_temp)

    area_crop = np.array(area_crop)

    hdr_cube = fits.getheader(path_cube)
    data_cube = fits.getdata(path_cube)

    if(np.all(area_crop!=0)):
        from astropy.wcs import WCS

        xi, xf, yi, yf = area_crop

        def remove_hdrs(hdrs, dim):
            names = np.array(['CDELT', 'CROTA', 'CRVAL', 'CTYPE', 'CUNIT', 'CRPIX', 'NAXIS'])
            for name in names:
                hdr = '{}{}'.format(name,dim)
                hdrs[hdr] = 'weee'
                hdrs.remove(hdr)

            return hdrs

        # REMOVE UNNECESSARY 4TH DIMENSION HDR IN CUBE
        # -> CAUSES TROUBLE IN WCS SLICING
        hdr_cube_rmhdr = fits.getheader(path_cube)
        remove_hdrs(hdr_cube_rmhdr, '4')

        print('CROPPING CUBE')
        wcs_cube = WCS(hdr_cube_rmhdr)
        wcs_cube_crop = wcs_cube[:,yi:yf,xi:xf]
        data_cube_crop = data_cube[:,yi:yf,xi:xf]

        hdr_cube_rmhdr.update(wcs_cube_crop.to_header())

        path_cube_cropped = os.path.splitext(path_cube)[0]+'_cropped.fits'
        fits.writeto(path_cube_cropped, data_cube_crop, hdr_cube_rmhdr, overwrite=True)

        hdr_cube = fits.getheader(path_cube_cropped)
        data_cube = fits.getdata(path_cube_cropped)

        path_cube = path_cube_cropped


    size = size_kernel

    zs = np.array(range(hdr_cube['NAXIS3']))

    pool = multiprocessing.Pool(processes=n_cores)

    print('SMOOTHING (n_cores={})'.format(n_cores))

    with tqdm(total=len(zs)) as pbar:
        for _ in tqdm(pool.imap_unordered(task_multirun, zs)):
            pbar.update()

    pool.close()
    pool.join()

    temps = natsorted(glob.glob(path_temp+"/*.npy"))

    print('REJOINING')

    smoothed = np.zeros_like(data_cube)

    for temp in tqdm(temps, total=len(temps)):

        z = np.load(temp)
        chan = int(os.path.basename(temp).split('.npy')[0])

        smoothed[chan,:,:] = z

    print('WRITING OUTPUTS')

    residual = data_cube - smoothed

    filename_smoothed = os.path.splitext(os.path.basename(path_cube))[0]+'_medfilt{}_smoothed.fits'.format(size_kernel)
    filename_residual = os.path.splitext(os.path.basename(path_cube))[0]+'_medfilt{}_residual.fits'.format(size_kernel)

    # print(filename_smoothed)
    fits.writeto(wdir+"/"+filename_smoothed, smoothed, hdr_cube, overwrite=True)
    fits.writeto(wdir+"/"+filename_residual, residual, hdr_cube, overwrite=True)


# EDIT HERE ===============================================================================
# wdir = '/media/cusped03/SDC3/mskim/residual/natural/'

# path_cube = wdir+'ZW3.msn_image_pbcorr.fits'
# run_main(n_cores=40, path_cube=path_cube, size_kernel=31, area_crop=[512,1536,512,1536])  
# 
wdir = '/media/cusped03/SDC3/mskim/residual/uniform/'

path_cube = wdir+'ZW3.msw_image_pbcorr.fits'
run_main(n_cores=40, path_cube=path_cube, size_kernel=9, area_crop=[512,1536,512,1536])      
# =========================================================================================
