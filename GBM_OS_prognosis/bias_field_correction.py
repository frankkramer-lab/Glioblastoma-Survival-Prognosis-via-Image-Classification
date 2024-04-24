#==============================================================================#
#  Function:     Bias field correction                                         #
#                    - saves the bias field corrected images                   #
#                    - need to be done before pipelines are executed           #
#                    - (uses N4 bias field correction from simpleITK)          #
#                                                                              #
#  copied and changed from https://github.com/Angeluz-07/MRI-preprocessing-techniques/blob/main/notebooks/03_bias_field_correction.ipynb   
#==============================================================================#

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import SimpleITK as sitk

#-----------------------------------------------------#
#                Bias Field Correction                #
#-----------------------------------------------------#
"""
The function bias_field_corr executes a bias field correction of the 
3D volume images in the specified path.
In this function we will iterate over all images in the given path and 
his subdirectories.

Expected structure: 
    images_dir/                     # image_dir_path = "dataset/images_dir"
        FLAIR/                      # modality = ["FLAIR", "T1", "T1CE", "T2"]
            sample001_flair.nii
            sample002_flair.nii
            ...
            sample050_flair.nii
        T1/                         # Directory / modality names
            sample001_t1.nii
            sample002_t1.nii
            ...
            sample050_t1.nii
        T1CE/
            sample001_t1ce.nii      # Sample names (indicies) should be unique!
            sample002._t1cenii
            ...
            sample050_t1ce.nii
        T2/
            sample001_t2.nii        # Sample names (indicies) should be unique!
            sample002_t2.nii
            ...
            sample050_t2.nii

Input patameters: 
    image_dir_path:     Path to the directory containing the folders for each modality.
    modalities:         List with the names of the folders containing the images 
                        (should be the modalities; lower or upper case). Default = None
    path_parent_result: Path where the results are stored. If None is provided it will use 
                        the image_dir_path. Default = None
"""
def bias_field_corr(image_dir_path, modalities, path_parent_result=None):
    # Define result path
    if path_parent_result == None:
        path_bfc = os.path.join(image_dir_path, "bias_field_corrected") 
    else: 
        path_bfc = path_parent_result
    if not os.path.exists(path_bfc): os.mkdir(path_bfc)

    # Iterate over each sample
    for modality in modalities: 
        # Define input path for specific modality
        path_modality = os.path.join(image_dir_path, modality)

        # Getting a sorted index list of all sample names for the modality
        index_list = sorted(os.listdir(path_modality))

        for sample_index in range(0, len(index_list)):
            # Define input path to specific sample
            path_img = os.path.join(path_modality, index_list[sample_index])
            # Load image via the SimpleITK package
            image_sample = sitk.ReadImage(path_img)

            # Create Head Mask to delimit ROI in images
            transformed = sitk.RescaleIntensity(image_sample, 0, 255)
            transformed = sitk.LiThreshold(transformed, 0, 1)
            head_mask = transformed

            # Bias field Correction
            # Get a shrinked version of the image and the head mask to reduce computationally costs
            shrink_factor = 4
            input_image = sitk.Shrink(image_sample, [shrink_factor] * image_sample.GetDimension())
            mask_image = sitk.Shrink(head_mask, [shrink_factor] * input_image.GetDimension())

            # Get a N4BiasFieldCorrectionImageFilter object to operate the correction
            bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
            # execute to correction with the shrinked versions
            corrected = bias_corrector.Execute(input_image, mask_image)

            # Get the original resolution
            log_bias_field = bias_corrector.GetLogBiasFieldAsImage(image_sample)
            corrected_image_full_resolution = image_sample / sitk.Exp( log_bias_field )

            # Save the image
            filename = index_list[sample_index]
            path_images_output = os.path.join(path_bfc, modality)
            if not os.path.exists(path_images_output): os.mkdir(path_images_output)
            sitk.WriteImage(corrected_image_full_resolution, os.path.join(path_images_output, filename))