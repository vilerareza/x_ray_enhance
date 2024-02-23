import os
import cv2 as cv
import numpy as np
import math
from skimage import exposure
from skimage.exposure import equalize_adapthist
import pandas as pd

from tqdm import tqdm



# Get upper limit using IQR
def get_outlier_thres(df, col_name, th1=0.25, th3=0.75, multiplier = 1.5):
    quart1 = df[col_name].quantile(th1)
    middle = df[col_name].quantile(0.5)
    quart3 = df[col_name].quantile(th3)
    iqr = quart3 - quart1
    lower = quart1 - multiplier * iqr
    upper = quart3 + multiplier * iqr
    print('quarts', quart1, quart3)
    return lower, upper, middle


# Histogram equalization
def equalize_hist_clahe(img, clahe_grid_size, clahe_clip, lower_limit, upper_limit, dtype):
    # Perform CLAHE
    img = equalize_adapthist(img, 
                            (int(img.shape[1]*clahe_grid_size), int(img.shape[0]*clahe_grid_size)), 
                            clip_limit = clahe_clip)
    # Rescale intensity
    img = exposure.rescale_intensity(img, 
                                    in_range=(np.min(img), np.max(img)), 
                                    out_range = (lower_limit, upper_limit))
    return img.astype(dtype)


def DarkChannel(im,sz):
    b,g,r = cv.split(im)
    dc = cv.min(cv.min(r,g),b)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(sz,sz))
    dark = cv.erode(dc,kernel)
    return dark


def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission


def Guidedfilter(im,p,r,eps):
    mean_I = cv.boxFilter(im,cv.CV_64F,(r,r))
    mean_p = cv.boxFilter(p, cv.CV_64F,(r,r))
    mean_Ip = cv.boxFilter(im*p,cv.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv.boxFilter(im*im, cv.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv.boxFilter(a,cv.CV_64F,(r,r))
    mean_b = cv.boxFilter(b,cv.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q


def TransmissionRefine(im,et):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t


def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res


# in / out files
input_path = 'images/chest.tiff'
out_dir = 'outputs/output_4'

# Inversion
invert = False

# Outlier factor
outlier_factor = 1.5

# Equalization params
clahe_limit = 1000
clahe_grid_size = 0.08
clahe_clip = 0.008

# Output range
out_lower = 0.1
out_upper = 0.9

# Mask dilation
mask_kernel_size = 25


if __name__ == '__main__':

    # Check if the given path is a directory or file
    is_single_file = False
    if os.path.isdir(input_path):
        # A Directory. Get list of files
        files = os.listdir(input_path)
    else:
        # Single file
        is_single_file = True
        files = [input_path]

    # Create the output directory if not present
    os.makedirs(out_dir, exist_ok=True)


    '''Processing'''

    for file in tqdm(files):

        try:

            '''Preprocess'''

            # Input image path
            if not is_single_file:
                img_path = os.path.join(input_path, file)
            else:
                img_path = file

            print (f'Processing {img_path}')

            # # Reading file
            # src = cv2.imread(fn);

            # I = src.astype('float64')/255;
        
            # dark = DarkChannel(I,15);
            # A = AtmLight(I,dark);
            # te = TransmissionEstimate(I,A,15);
            # t = TransmissionRefine(src,te);
            # J = Recover(I,t,A,0.1);
            # J=J*255

            img_ori = cv.imread(img_path, -1)
            if len(img_ori.shape) > 2:
                if img_ori.shape[2] == 4:
                    img_ori = cv.cvtColor(img_ori, cv.COLOR_BGRA2GRAY)
                elif img_ori.shape[2] == 3:
                    img_ori = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
            
            # Getting dtype
            dtype = img_ori.dtype
            print(f'dtype: {img_ori.dtype}')
            
            # Initial normalization
            img_ori = (img_ori/np.max(img_ori))*np.iinfo(dtype).max
            img_ori = img_ori.astype(dtype)

            # Inversion
            if invert:
                img_ori = np.iinfo(dtype).max - img_ori  

            img = img_ori.copy()

            # Center patch
            img_patch = img[int(img.shape[0]/4):-int(img.shape[0]/4),
                            int(img.shape[1]/4):-int(img.shape[1]/4)]

            # Getting properties
            mean_val = int(np.mean(img))
            min_val = np.min(img)
            max_val = np.max(img)
            print (f'Original intensity (min, max, mean): {min_val}, {max_val}, {mean_val}')


            '''Intensity Rescale'''

            # Find intensity distribution
            img_flat = img_patch.copy().flatten()
            img_df = pd.DataFrame(img_flat)
            lower, upper, middle = get_outlier_thres(img_df, 
                                                    0, 
                                                    multiplier=outlier_factor)
            lower = max(0, lower)
            upper = min(max_val, upper)
            print (f'Intensity distribution (lower, upper, middle):, {lower}, {upper}, {middle}')

            # Intensity rescale
            img = exposure.rescale_intensity(img, in_range=(lower, upper))

            # Denoise
            # blur_level = 25
            # img = cv.GaussianBlur(img, (blur_level, blur_level), 0)

            '''Histogram equalization'''
            lower_limit = int(out_lower * np.iinfo(dtype).max)
            upper_limit = int(out_upper * np.iinfo(dtype).max)
            img = equalize_hist_clahe(img, 
                                      clahe_grid_size,
                                      clahe_clip,
                                      lower_limit,
                                      upper_limit,
                                      dtype
                                      )

            # Path for output
            out_file = os.path.join(out_dir, f'output_{os.path.splitext(os.path.split(file)[-1])[0]}.png')

            cv.imwrite(out_file, img)
        

        except Exception as e:
           print (f'{e}: Error processing {file}. Check if it is valid image file')