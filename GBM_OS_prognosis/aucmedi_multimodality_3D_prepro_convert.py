#==============================================================================#
#  Function:     Image prerprocessing                                          #
#                    - based on DataGenerators Subfunction execution           #
#                    - apply suubfunctions before converting to big 3D images  #
#                    - save preprocessed images seperately in                  #
#                      path_result/preprocessed_images                         #
#  Function:      Concatenate/Convert to big 3D images                         #
#                    - structure: [[FLAIR, T1]                                 #
#                                  [T1CE,  T2]]                                #
#                    - sort all files in the direcotries the same way          #
#                    - check if all directories include the same number        #
#                    - iterate over all samples and concate them to big 3D     #
#                      image if they belong to the same sample                 #
#==============================================================================#

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import numpy as np
from aucmedi.data_processing.io_loader import sitk_loader
import SimpleITK as sitk

#-----------------------------------------------------#
#                 Image Preprocessing                 #
#-----------------------------------------------------#
"""
The function preprocess_image_via_subfunc transforms each sample with 
all subfunctions, defined in a list as input parameter. 
In this function we will iterate over all images in the given path and 
his subdirectories.
No detailed check for the existence of a sample in all modalities is 
carried out here. This is done in the converter function run_großes_3D_converter.

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
                        (should be the modalities; lower or upper case). 
    subfunctions:       List of Subfunctions class instances which will be SEQUENTIALLY 
                        executed on the data set.
    resampling:         Tuple of 3x floats with z,y,x mapping encoding voxel spacing. 
                        If passing None, no normalization will be performed.
    path_result:        Path where the results are stored. 

                        If None is provided it will store the preprocessed images in 
                        ./preprocessed_images and in ./big_3D_image_dataset the converted images 
                        (if converting = True). 

                        If a path is provided and converting = True, it will store the preprocessed 
                        images in path_result/preprocessed_images the converted images in 
                        path_result/big_3D_image_dataset. 

                        Default = None
    converting          Boolean, weather run_großes_3D_converter (converting to big 3D images) 
                        should be run automatically. Default = True

Return: 
    result_path:        Weather the path to the preprocessed images (converting = False) or 
                        to the converted big 3D images (converting = True).
"""
def preprocess_image_via_subfunc(image_dir_path, modalities, subfunctions, resampling, 
                                 path_result=None, converting=True):

    return_paths = []

    # Result directory
    if path_result == None:
        path_results_preprocessed = "preprocessed_images" 
    else:
        if converting:
            path_results_preprocessed = os.path.join(path_result, "preprocessed_images")
        else:
            path_results_preprocessed = path_result
    if not os.path.exists(path_results_preprocessed) : os.mkdir(path_results_preprocessed)

    for modality in modalities: 
        # Define input path for specific modality
        path_images_input = os.path.join(image_dir_path, modality)

        # Getting a sorted index list of all sample names for the modality
        index_list = sorted(os.listdir(path_images_input))

        for sample_index in range(0, len(index_list)):
            # Get image as numpy-array
            image_array = sitk_loader(index_list[sample_index], 
                                      path_imagedir=path_images_input, 
                                      image_format=None, 
                                      grayscale=True, 
                                      resampling = resampling)
            # Apply subfunctions on image
            for sf in subfunctions:
                image_array = sf.transform(image_array)
            # save image with same name in path_results_preprocessed 
            filename = index_list[sample_index] 
            path_images_output = os.path.join(path_results_preprocessed, modality)
            if not os.path.exists(path_images_output): os.makedirs(path_images_output)
            image = sitk.GetImageFromArray(image_array)
            sitk.WriteImage(image, os.path.join(path_images_output, filename))
    
        # Append path_images_output (preprocessed_images/path_modality_1/) to return_paths list 
        return_paths.append(path_images_output)

    if converting: 
        # Convert/Concatenate images to multimodality 3D image
        path_res = os.path.join(path_result, "big_3D_image_dataset")
        path_3D = run_multimodality_3D_converter(path_results_preprocessed, 
                                      modalities=modalities, 
                                      resampling=None, 
                                      path_result=path_res)
        return path_3D
    else:
        return return_paths

#-----------------------------------------------------#
#                 Image 3D Converter                  #
#-----------------------------------------------------#
"""
The function run_multimodality_3D_converter convertes/combines different 
modalities of 3D Nifti images to a big 3D image with the modalities. 
In this function we will iterate over all images in the given path and 
his subdirectories.

Use the preprocessed pictures (preprocess_image_via_subfunc) of the 
images. The subfunctions for preparing can be defined. 

structure of big 3D image: example for ["FLAIR", "T1", "T1CE", "T2"]
    [[FLAIR, T1]
    [T1CE,  T2]]

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
    image_dir_path:     Path to the directory containing the folders for each modality to the 
                        preprocessed images.
    modalities:         List with the names of the folders containing the images (should be the 
                        modalities; lower or upper case). 
    resampling:         Tuple of 3x floats with z,y,x mapping encoding voxel spacing. If passing
                        None, no normalization will be performed.
    path_result:        Path where the results are stored. If None is provided it will store 
                        them in ./big_3D_image_dataset. Default = None

Return: 
    result_path:        Path to the converted big 3D images (converting = True)
"""
def run_multimodality_3D_converter(image_dir_path, modalities, resampling=None, path_result=None):
    # Result directory
    if path_result == None:
        path_big_3D = "big_3D_image_dataset" 
    else:
        path_big_3D = path_result
    if not os.path.exists(path_big_3D) : os.mkdir(path_big_3D)

    # Sort all directories 
    # Iterate over all images
    for modality in modalities: 
        # Get the path to the images of the specified modality
        path_modality = os.path.join(image_dir_path, modality)

        index_list = sorted(os.listdir(path_modality))
        format = None

        # Identify image format by peaking first image
        if format is None:
            format = index_list[0].split(".")[-2] + index_list[0].split(".")[-1]
            # Raise Exception if image format is not Nifti .gz
            if format.lower() != "niigz":
                raise Exception("Must have an Nifti format as follows: nii.gz .", path_modality)
        
        # Get modality without other strings like "images_"flair
        if modality.find("_") != -1: modality_from_path = modality.split("_")[-1] 
        else: modality_from_path = modality  

        # Getting a sorted index list for each modality       
        if modality_from_path.lower() == "flair" : 
            # Sorted index list for flair images
            list_flair = index_list 
            # Path to the flair images (needed for the sitk_loader)
            path_flair = path_modality 
        elif modality_from_path.lower() == "t1" : 
            list_t1 = index_list
            path_t1 = path_modality
        elif modality_from_path.lower() == "t1ce" : 
            list_t1ce = index_list
            path_t1ce = path_modality
        elif modality_from_path.lower() == "t2" : 
            list_t2 = index_list
            path_t2 = path_modality
        else: 
            print("Something went horrible wrong by sorting the input folders.")
            exit()
        
    # Check if all directories include the same number of samples
    len_flair = len(list_flair)
    len_t1 = len(list_t1)
    len_t1ce = len(list_t1ce)
    len_t2 = len(list_t2)
    if len_flair == len_t1 == len_t2 == len_t1ce: 
        print("The number of files in each modality is: {}".format(len_flair))
    else: 
        raise Exception("""There are different number of samples in the modalities. 
                (Number of modalities: FLAIR, T1, T1ce, T2)""", len_flair, len_t1, len_t1ce, len_t2)

    # Iterate over all samples and concatenate them to 3D image 
    for sample_index in range(0, len_flair):
        sample_split = list_flair[sample_index].split("_")
        # the last two list elements are [modality, normalized.nii.gz]
        sample_without_modality_split = sample_split[: len(sample_split) - 2] 
        sample_without_modality = "_".join(sample_without_modality_split)

        # Check if the images belong to the same sample 
        if list_t1[sample_index].startswith(sample_without_modality) and \
            list_t1ce[sample_index].startswith(sample_without_modality) and \
            list_t2[sample_index].startswith(sample_without_modality): 
            # Get image as numpy-array
            flair_image = sitk_loader(list_flair[sample_index], 
                                    path_imagedir=path_flair,
                                    image_format=None, 
                                    grayscale=True, 
                                    resampling = resampling)
            t1_image = sitk_loader(list_t1[sample_index], 
                                    path_imagedir=path_t1, 
                                    image_format=None, 
                                    grayscale=True, 
                                    resampling = resampling)
            t1ce_image = sitk_loader(list_t1ce[sample_index], 
                                    path_imagedir=path_t1ce, 
                                    image_format=None, 
                                    grayscale=True, 
                                    resampling = resampling)
            t2_image = sitk_loader(list_t2[sample_index], 
                                    path_imagedir=path_t2, 
                                    image_format=None, 
                                    grayscale=True, 
                                    resampling = resampling)
            # Concatenate 4 images (same sample of each modality)
            flair_t2 = np.concatenate((flair_image, t2_image), axis = 1)
            t1_t1ce = np.concatenate((t1_image, t1ce_image), axis = 1)
            big_3D = np.concatenate((flair_t2, t1_t1ce), axis = 2) 
            # Store image as nii file
            filename = sample_without_modality + ".nii" 
            nifti_image = sitk.GetImageFromArray(big_3D)
            sitk.WriteImage(nifti_image, os.path.join(path_big_3D, filename))
    
    return path_big_3D  