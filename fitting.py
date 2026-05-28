import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Tuple
from functools import partial # Use when changing which parameters are fit with the same fit function

# Called to fit each vortex after initial identification. 
# The monopole method is known to not always be the most applicable, modify as needed. If the fitting funtion is changed, be sure to change the header
# in the other files as needed

def robust_normalize(img: np.ndarray, p: Tuple[float, float] = (1, 99)) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0, 1).astype(np.float32)

def monopole_fit_z(xy, xc, yc, z):
    # Different functions/fit parameters may be more applicable
    x = xy[:,0] 
    y = xy[:,1] 
    denom = (((x-xc)**2 + (y-yc)**2) + z**2)**(3/2)

    return z**3 / denom # Peak height will be 1, so A=z^2

def disc_points(xc,yc,r,img):
    # Returns all points in img that are within r distance away from the points(xc,yc)
    x_min = int(np.max((np.floor(xc-r),0)))
    x_max = int(np.min((np.ceil(xc+r),np.shape(img)[1]-1)))
    y_min = int(np.max((np.floor(yc-r),0)))
    y_max = int(np.min((np.ceil(yc+r),np.shape(img)[0]-1)))
    points = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            dist_sq = (x - xc) ** 2 + (y - yc) ** 2
            if  dist_sq <= r ** 2 and dist_sq != 0:
                points.append((x, y))
    return np.array(points)

def fit_monopole(img, xc, yc, r, plot=False): 
    # Fits a sub image containing a disc of radius r around point (xc,yc). Omits zeros from the fit. Can also plot the sub image and fit if needed. 
    xy = disc_points(xc, yc, r, img)
    mask_indices = []
    for i, (x, y) in enumerate(xy):
        if img[y,x]==0:
            mask_indices.append(i)
    xy = np.delete(xy, mask_indices, 0)
    x = xy[:,0]
    y = xy[:,1]
    
    params, pcov = curve_fit(monopole_fit_z, xy, robust_normalize(img[y, x]), p0=[xc, yc, r])
    pstd = np.sqrt(np.diag(pcov))

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        sub_img = robust_normalize(img[np.min(y):np.max(y), np.min(x):np.max(x)])   
        im0 = ax[0].imshow(sub_img, origin='lower', extent = [np.min(x), np.max(x), np.min(y), np.max(y)], vmin=0, vmax=1)
        ax[0].set_title("Normalized Image")
        ax[0].set_xlabel("x (px)")
        ax[0].set_ylabel("y (px)")
        cbar = fig.colorbar(im0, ax=ax[0])
        cbar.set_label('Normalized B Field')
        
        fit_img = np.zeros_like(img)
        
        for i in range(np.shape(img)[1]):
            for j in range(np.shape(img)[0]):
                fit_img[j,i] = monopole_fit_z(np.array([(i,j)]), *params)[0]
        print(params, pstd)
        sub_fit_img = fit_img[np.min(y):np.max(y), np.min(x):np.max(x)]
        im1 = ax[1].imshow(sub_fit_img, origin='lower', extent = [np.min(x), np.max(x), np.min(y), np.max(y)], vmin=0, vmax=1)
        ax[1].set_title("Vortex Monopole Fit")
        ax[1].set_xlabel("x (px)")
        ax[1].set_ylabel("y (px)")
        cbar = fig.colorbar(im1, ax=ax[1])
        cbar.set_label('Normalized B Field')
        plt.show()
    
    return params, pstd