from json.tool import main
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


# $ git clone https://github.com/SuperElastix/SimpleElastix
# $ mkdir build
# $ cd build
# $ cmake ../SimpleElastix/SuperBuild
# $ make -j4

# set default parameter
new_spacing = [1, 1, 1]
registrated_type = 'affine'
# See the Dipy registration tutorial for the details of what these parameters mean:
nbins = 32
sampling_prop = None
level_iters = [10, 10, 5]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

dirname = r'F:\BaiduNetdiskDownload\shuju\shuju\226630'

filename = r'F:\BaiduNetdiskDownload\shuju\shuju\226630\ep2d_diff_tra_b800_b1000_p2_160 - as a 3 frames MultiVolume by Siemens.B-value.raw.gz'
filename2 = r'F:\BaiduNetdiskDownload\shuju\ep2d_diff_tra_b800_b1000_p2_160 - as a 3 frames MultiVolume by Siemens.B-value.raw.gz'
def registrate_valid():
    moving = nib.load(r'F:\BaiduNetdiskDownload\shuju\shuju\226630\8 ep2d_diff_tra_b800_b1000_p2_160_ADC_1_resampled.nii')
    fixed = nib.load(r'F:\BaiduNetdiskDownload\shuju\shuju\226630\4 t2_tse_tra_p2_4_resampled.nii')
    moving_data = moving.get_data()
    fixed_data = fixed.get_data()
    moving_affine = moving.affine
    fixed_affine = fixed.affine
    identity = np.eye(4)
    affine_map = AffineMap(identity, fixed_data.shape, fixed_affine, moving_data.shape, moving_affine)
    resampled = affine_map.transform(moving_data)  # 3-4 minutes
    # regtools.overlay_slices(fixed_data, resampled, None, 0, 'fixed', 'moving')
    # regtools.overlay_slices(fixed_data, resampled, None, 1, 'fixed', 'moving')
    # regtools.overlay_slices(fixed_data, resampled, None, 2, 'fixed', 'moving')
    metric = MutualInformationMetric(nbins, sampling_prop)
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
    
    # optimize the translations
    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(fixed_data, moving_data, transform, params0, fixed_affine, moving_affine)
    # translation.affine
    transformed = translation.transform(moving_data)
    # regtools.overlay_slices(fixed_data, transformed, None, 0, 'fixed', 'transformed')
    # regtools.overlay_slices(fixed_data, transformed, None, 1, 'fixed', 'transformed')
    # regtools.overlay_slices(fixed_data, transformed, None, 2, 'fixed', 'transformed')
    
    # optimize a rigid-body transform
    transform = RigidTransform3D()
    rigid = affreg.optimize(fixed_data, moving_data, transform, params0, fixed_affine, moving_affine, starting_affine=translation.affine)
    transformed = rigid.transform(moving_data)
    
    # full affine registration
    transform = AffineTransform3D()
    affreg.level_iters = [1000, 1000, 100]
    affine = affreg.optimize(fixed_data, moving_data, transform, params0, fixed_affine, moving_affine, starting_affine=rigid.affine)
    transformed = affine.transform(moving_data)
    regtools.overlay_slices(fixed_data, transformed, None, 0, 'fixed', 'transformed')
    regtools.overlay_slices(fixed_data, transformed, None, 1, 'fixed', 'transformed')
    regtools.overlay_slices(fixed_data, transformed, None, 2, 'fixed', 'transformed')
    transformed_image = sitk.GetImageFromArray(transformed.transpose(2,1,0))
    resampled_fix = sitk.ReadImage(r'F:\BaiduNetdiskDownload\shuju\shuju\226630\4 t2_tse_tra_p2_4_resampled.nii')
    transformed_image.SetOrigin(resampled_fix.GetOrigin())
    transformed_image.SetSpacing(resampled_fix.GetSpacing())
    transformed_image.SetDirection(resampled_fix.GetDirection())
    sitk.WriteImage(transformed_image, r'F:\BaiduNetdiskDownload\shuju\shuju\226630\8 ep2d_diff_tra_b800_b1000_p2_160_ADC_1_registrated.nii')
    

# write a function for z-score
def z_score_process_image(sitk_image):
    sitk_arr = sitk.GetArrayFromImage(sitk_image)  # ndarray 
    volume = sitk_arr != 0  # type bool
    sitk_arr[volume] = (sitk_arr[volume] - np.mean(sitk_arr[volume])) / np.std(sitk_arr[volume])
    return sitk.GetImageFromArray(sitk_arr)

def resample_process_image(sitk_image, new_spacing=[1,1,1]):
    ori_size = np.array(sitk_image.GetSize(), dtype=np.int)
    ori_origin = sitk_image.GetOrigin()
    ori_spacing = np.array(sitk_image.GetSpacing(), dtype=np.float)
    ori_direction = sitk_image.GetDirection()
    
    new_spacing = np.array(new_spacing, dtype=np.float)
    new_size = ori_size * (ori_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays
    
    resampler_filter = sitk.ResampleImageFilter()
    resampler_filter.SetReferenceImage(sitk_image)
    resampler_filter.SetSize(new_size)
    resampler_filter.SetInterpolator(sitk.sitkBSpline)
    resampler_filter.SetOutputSpacing(new_spacing)
    resampler_filter.SetOutputOrigin(ori_origin)
    resampler_filter.SetOutputDirection(ori_direction)
    resampled_sitk_image = resampler_filter.Execute(sitk_image)
    
    return resampled_sitk_image

def registration_process_image(fixed_image, moving_image, type='affine'):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()
    registed_image = elastixImageFilter.GetResultImage()
    return registed_image
                                      

def z_score_resample():
    under_preprocess = {}
    for file in os.listdir(dirname):
        if 'zscore' in file:
            continue
        if 'resampled' in file:
            continue
        if 't2' in file:
            under_preprocess['T2WI'] = os.path.join(dirname, file)
        if 'ADC' in file:
            under_preprocess['ADC'] = os.path.join(dirname, file)
        if 'DWI' in file:
            under_preprocess['DWI'] = os.path.join(dirname, file)
    # Step1: z-score each image
    for key, value in under_preprocess.items():
        z_score_name = value.split('.nii')[0] + '_zscore.nii'
        if not os.path.exists(z_score_name):
            image = sitk.ReadImage(value)
            image = sitk.Cast(image, sitk.sitkFloat64)
            original_origin = image.GetOrigin()
            original_spacing = image.GetSpacing()
            original_direction = image.GetDirection()
            z_score_image = z_score_process_image(image)
            z_score_image.SetOrigin(original_origin)
            z_score_image.SetSpacing(original_spacing)
            z_score_image.SetDirection(original_direction)
            sitk.WriteImage(z_score_image, z_score_name)
    # Step2: Resample each image
    for key, value in under_preprocess.items():
        z_score_name = value.split('.nii')[0] + '_zscore.nii'
        resampled_name = value.split('.nii')[0] + '_resampled.nii'
        if os.path.exists(z_score_name) and not os.path.exists(resampled_name):
            z_score_image = sitk.ReadImage(z_score_name)
            resampled_image = resample_process_image(z_score_image)
            sitk.WriteImage(resampled_image, resampled_name)
    # Step3: Registration (Affine)
    fixed_name = under_preprocess['T2WI'].split('.nii')[0] + '_resampled.nii'
    fixed_image = sitk.ReadImage(fixed_name)
    # fixed = under_preprocess.pop(under_preprocess['T2WI'])
    for key, value in under_preprocess.items():
        resampled_name = value.split('.nii')[0] + '_resampled.nii'
        regis_name = value.split('.nii')[0] + '_registed.nii'
        if os.path.exists(resampled_name) and not os.path.exists(regis_name):
            if key == 'T2WI':
                continue
            moving_image = sitk.ReadImage(resampled_name)
            registed_image = registration_process_image(fixed_image, moving_image)
            sitk.WriteImage(registed_image, regis_name)
    
    
        

if __name__ == '__main__':
    # z_score_resample()
    registrate_valid()
    
    

    # readdata, header = nrrd.read(filename2)
    # b = 1