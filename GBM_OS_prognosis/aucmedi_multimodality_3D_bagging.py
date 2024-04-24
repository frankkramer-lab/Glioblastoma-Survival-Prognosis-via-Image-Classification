#==============================================================================#
#  Overview:                                                                   #
#        architectures: ResNet50                                               #
#        3D images (multimodality)                                             #
#        resampling (1.5,1.5,1.5)                                              #
#        Subfunctions: Bias field correction, Standardize, Padding, Cropping   #
#                      and Chromer                                             #
#        transfer learning                                                     #
#        batch size = 4                                                        #
#        model.tf_epochs = 10                                                  #
#==============================================================================#

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import pandas as pd
import argparse
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score
import tensorflow as tf
# AUCMEDI libraries
from aucmedi import *
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi.utils.class_weights import compute_multilabel_weights, compute_class_weights
from aucmedi.data_processing.subfunctions import *
from aucmedi.sampling.split import sampling_split
from aucmedi.neural_network.loss_functions import *
from aucmedi.evaluation import *
from aucmedi.utils.class_weights import compute_class_weights, compute_multilabel_weights
from aucmedi.ensemble.bagging import *
#3D Converter-File
from aucmedi_multimodality_3D_prepro_convert import *
# Bias field correction
from bias_field_correction import *

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="AUCMEDI Training for ASH")
parser.add_argument("-r", "--results", help="Name of output directory",
                    required=False, type=str, dest="path_results", default="results")
parser.add_argument("-g", "--gpu", help="GPU ID selection for multi cluster",
                    required=False, type=int, dest="gpu", default=0)
args = parser.parse_args()

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
# Provide parent input path to the images we want to 
# preprocess and concatenate (original)
input_path = "/share/rhuh_survival_aucmedi/dataset_full"

# Provide names/paths of modalities 
path_images_ensemble = ["FLAIR",
                        "T1",
                        "T1CE",
                        "T2"]

# Provide path to the ground truth csv
path_labels = "data_survival.csv"

# Colum class
column_sample = "ID"
column_class = "survival_groups"

# Result/Model directory
path_results_parent = "/home/main/storage/big_3D/bagging"
path_results = os.path.join(path_results_parent, args.path_results)
if not os.path.exists(path_results) : os.makedirs(path_results)

# Define architecture which should be processed
architecture = "3D.ResNet50"

# Define some parameters
batch_size = 4
batch_queue_size = 10
processes = 6
threads = 4
k_fold = 5

# Define input shape
resampling = (1.5, 1.5, 1.5)
input_shape = (103, 320, 260) # (concatenated images)

# Define training parameter
epochs = 250
iterations = None
prepare_images = True # activate for more speed
multiprocessing = True # deactive under Windows

# Is the classification task a multi-label problem?
is_multilabel = False
activation_function = 'sigmoid' if is_multilabel else 'softmax'

#-----------------------------------------------------#
#              Setup of Tensorflow Stack              #
#-----------------------------------------------------#
# Set dynamic grwoth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Fix GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

#-----------------------------------------------------#
#                Bias Field Correction                #
#-----------------------------------------------------#
"Here we proceed the bias field correction."
# Configurations
# Provide modality names to imaging data for Bias field correction
path_images_bfc = [["FLAIR"], ["T1"], ["T1CE"], ["T2"]]

# bias field corrected output directory
path_bfc = "/home/main/storage/bias_field_corrected"
if not os.path.exists(path_bfc) : os.makedirs(path_bfc)
for modality in path_images_ensemble:
    path = os.path.join(path_bfc, modality)
    if not os.path.exists(path): os.mkdir(path)

# check each modality individually and execute the BFC for each modality indiually, 
# because it is computionally expensive. No redundant execution
if os.listdir(os.path.join(path_bfc, path_images_ensemble[0])) == []:
    bias_field_corr(input_path, path_images_bfc[0], path_parent_result=path_bfc)
if os.listdir(os.path.join(path_bfc,  path_images_ensemble[1])) == []:
    bias_field_corr(input_path, path_images_bfc[1], path_parent_result=path_bfc)
if os.listdir(os.path.join(path_bfc, path_images_ensemble[2])) == []:
    bias_field_corr(input_path, path_images_bfc[2], path_parent_result=path_bfc)
if os.listdir(os.path.join(path_bfc, path_images_ensemble[3])) == []:
    bias_field_corr(input_path, path_images_bfc[3], path_parent_result=path_bfc)

#-----------------------------------------------------#
#    Setup of Converting Image & input data reader    #
#-----------------------------------------------------#
"""
Here we converting the images to the big 3D images, if it isn't already done. 

We also load our images and the labels and splitting the samples beforehand, 
so we can iterate over each split and train the model.
"""
# Define Subfunctions
sf_list = [Standardize(mode="grayscale"), # is needed for the Chromer-subfunction
           Padding(mode="constant", shape=input_shape),
           Crop(shape=(103, 160, 130), mode="center"),
           Chromer(target="rgb")]

#checking if we need to convert the images
path_big_3D_output = "/home/main/storage/big_3D/dataset_bfc/big_3D_image_dataset" 
path_parent = "/home/main/storage/big_3D/dataset_bfc" # includes /preprocessed_images and /big_3D_image_dataset
if not os.path.exists(path_parent): os.makedirs(path_parent)
if not os.path.exists(path_big_3D_output) or len(os.listdir(path_big_3D_output)) != len(os.listdir(os.path.join(path_bfc, path_images_ensemble[0]))): 
    # Concatenate images
    path_big_3D_output = preprocess_image_via_subfunc(image_dir_path=path_bfc, 
                             modalities = path_images_ensemble,
                             subfunctions = sf_list, 
                             resampling=resampling,
                             path_result=path_parent, 
                             converting=True)

# Pillar #1: Initialize input data reader
ds = input_interface(interface="csv", 
                         path_data=path_labels,
                         path_imagedir=path_big_3D_output,
                         training=True,
                         col_sample = column_sample,
                         col_class = column_class)
(index_list, class_ohe, class_n, class_names, image_format) = ds

# Perform dataset exploration
evaluate_dataset(
    samples=None,
    labels=class_ohe,
    out_path=path_results,
    class_names=class_names,
    plot_barplot=True
)

# Perform sampling 
ds_train, ds_test = sampling_split(
    samples=index_list,
    labels=class_ohe,
    sampling=[0.80, 0.20],
    stratified=True,
    seed=111,
    iterative=False,
)

# Get sample names without test dataset 
index_list_folds = [x for x in index_list if x not in ds_test[0]]
indx_list_for_samples = [index_list.index(x) for x in index_list_folds]
# get ohe without those of the test dataset
ohe_folds = [class_ohe[i] for i in range(len(class_ohe)) if i in indx_list_for_samples]

# Perform sampling: get a 5-Fold data set of the ds_train set
fold_list = sampling_kfold(
    samples=index_list_folds,
    labels=ohe_folds,
    metadata=None,
    n_splits=k_fold,
    stratified=True,
    iterative=False,
    seed=111,
)

#-----------------------------------------------------#
#           AUCMEDI Classifier - Training             #
#-----------------------------------------------------#
""" 
    Here we build the AUCMEDI classifier for training.
    This is done dynamically and on the same way for each fold.

    The advantage is that we can clean up a lot of redundant/unnecessary code.
    The disadvantage is that each fold runs the same preprocessing, 
    which may be not ideal for some scenarios.
"""
def run_bagging_training(fold, fold_nr):
    # Initialize Volume Augmentation
    aug = BatchgeneratorsAugmentation(image_shape=input_shape,
                    mirror=True, rotate=True, scale=True,
                    elastic_transform=True, gaussian_noise=False,
                    brightness=True, contrast=True, gamma=False)

    # Loss
    if is_multilabel:
        class_weights = compute_multilabel_weights(ds_train[1])
        loss = multilabel_focal_loss(class_weights)
    else:
        cw_loss, class_weights = compute_class_weights(ds_train[1])
        loss = categorical_focal_loss(cw_loss)

    # Pillar #2: Initialize model
    model = NeuralNetwork(n_labels=class_n,
                         channels=3,
                         architecture=architecture,
                         input_shape=input_shape,
                         workers=processes,
                         batch_queue_size=batch_queue_size,
                         loss=loss,
                         metrics=[AUC(100),
                                 F1Score(num_classes=3, average="macro")],
                         pretrained_weights=True,
                         multiprocessing=multiprocessing,
                         activation_output=activation_function,
                         )
    model.tf_epochs = 10                  

    # Pillar #3: Initialize training and validation Data Generators
    train_gen = DataGenerator(fold[0], path_imagedir=path_big_3D_output,
                            labels=fold[1],
                            batch_size=batch_size,
                            data_aug= aug,
                            shuffle=True,
                            subfunctions=[],
                            resize=None,
                            standardize_mode=model.meta_standardize,
                            grayscale=False,
                            prepare_images=prepare_images,
                            sample_weights=None,
                            seed=0,
                            image_format=image_format,
                            workers=threads,
                            loader=sitk_loader,
                            resampling=None,
                            )
    val_model_gen = DataGenerator(fold[2], path_imagedir=path_big_3D_output,
                            labels=fold[3],
                            batch_size=batch_size,
                            data_aug=None,
                            shuffle=False,
                            subfunctions=[],
                            resize=None,
                            standardize_mode=model.meta_standardize,
                            grayscale=False,
                            prepare_images=prepare_images,
                            sample_weights=None,
                            seed=0,
                            image_format=image_format,
                            workers=threads,
                            loader=sitk_loader,
                            resampling=None,
                            )

    # Create model directory
    path_model_bagging = os.path.join(path_results, "models")
    if not os.path.exists(path_model_bagging) : os.mkdir(path_model_bagging)

    # Define callbacks
    cb_ml = ModelCheckpoint(os.path.join(path_model_bagging,
                                        architecture + "fold_nr_" + str(fold_nr) + ".model.best_loss.hdf5"),
                            monitor="val_loss", verbose=1,
                            save_best_only=True, mode="min")
    cb_ma = ModelCheckpoint(os.path.join(path_model_bagging,
                                        architecture + "fold_nr_" + str(fold_nr) + ".model.best_auc.hdf5"),
                            monitor="val_auc", verbose=1,
                            save_best_only=True, mode="max")
    cb_mr = ModelCheckpoint(os.path.join(path_model_bagging,
                                        architecture + "fold_nr_" + str(fold_nr) + ".model.best_f1.hdf5"),
                            monitor="val_f1_score", verbose=1,
                            save_best_only=True, mode="max")
    cb_cl = CSVLogger(os.path.join(path_model_bagging, "training.csv"),
                    separator=',', append=True)
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8,
                            verbose=1, mode='min', min_lr=1e-7)
    cb_es = EarlyStopping(monitor='val_loss', patience=32, verbose=1)
    callbacks = [cb_ml, cb_ma, cb_mr, cb_cl, cb_lr, cb_es]

    # Train model
    history = model.train(train_gen, val_model_gen, epochs=epochs, 
                            iterations=iterations,
                            callbacks=callbacks, transfer_learning=True)

    # Dump latest model
    model.dump(os.path.join(path_model_bagging, architecture + "fold_nr_" + str(fold_nr) + ".model.last.hdf5"))

    # Clear GPU cache to encourage garbage collection and avoid memory clutter
    tf.keras.backend.clear_session()

#-----------------------------------------------------#
#          AUCMEDI Classifier - Prediction            #
#-----------------------------------------------------#
""" 
    Here we build the AUCMEDI classifier for prediction.
    This is done dynamically and on the same way for each fold.

    The advantage is that we can clean up a lot of redundant/unnecessary code.
    The disadvantage is that each modality runs the same preprocessing, 
    which may be not ideal for some scenarios.

    model_type parameter: Options ["best_loss", "last", "best_f1", "best_auc"]
"""
def run_prediction(fold_nr, model_type="best_loss"):
    # Pillar #2: Initialize model
    model = NeuralNetwork(n_labels=class_n,
                         channels=3,
                         architecture=architecture,
                         input_shape=input_shape,
                         workers=processes,
                         batch_queue_size=batch_queue_size,
                         multiprocessing=False,
                         activation_output=activation_function,
                         )

    # Load weights
    path_model_3D = os.path.join(path_results, "models")
    path_model = os.path.join(path_model_3D, architecture + "fold_nr_" + str(fold_nr) + ".model." + model_type + ".hdf5")
    model.model.load_weights(path_model)

    # Pillar #3: Initialize Data Generators for the selected subset
    datagen = DataGenerator(ds_test[0], path_imagedir=path_big_3D_output,
                            labels=None,
                            image_format=image_format,
                            subfunctions=[],
                            batch_size=batch_size,
                            resize=None,
                            standardize_mode=model.meta_standardize,
                            data_aug=None,
                            shuffle=False,
                            grayscale=False,
                            sample_weights=None,
                            workers=threads,
                            prepare_images=False,
                            loader=sitk_loader,
                            seed=0,
                            resampling=None,
                            )

    # Create prediction directory
    path_pred_dir = os.path.join(path_results, "predictions")
    if not os.path.exists(path_pred_dir) : os.mkdir(path_pred_dir)

    # Compute predictions
    preds = model.predict(datagen)

    # Store predictions to disk
    df_index = pd.DataFrame(data={"index": ds_test[0]})
    df_pd = pd.DataFrame(data=preds, columns=["pd_" + i for i in class_names])
    df_gt = pd.DataFrame(data=ds_test[1], columns=["gt_" + i for i in class_names])
    df_merged = pd.concat([df_index, df_pd, df_gt], axis=1, sort=False)
    path_preds = os.path.join(path_pred_dir, 
                            "preds." + architecture + "fold_nr_" + str(fold_nr) + "." + model_type + "." + "test" + ".csv")
    df_merged.to_csv(path_preds, index=False)

    # Create evaluation directory
    path_eval_dir = os.path.join(path_results, "evaluations")
    if not os.path.exists(path_eval_dir) : os.mkdir(path_eval_dir)

    # Evaluate performance
    evaluate_performance(
        preds=preds,
        labels=ds_test[1],
        out_path=path_eval_dir,
        class_names=class_names,
        suffix=architecture + "fold_nr_" + str(fold_nr) + "." + model_type + "." + "test",
    )

    # Clear GPU cache to encourage garbage collection and avoid memory clutter
    tf.keras.backend.clear_session()

    # Return index & prediction list
    return ds_test[0], preds, ds_test[1]

#-----------------------------------------------------#
#          Ensembler (Aggregate) - Prediction         #
#-----------------------------------------------------#
def run_bagging_prediction(model_type):
    # Iterate over each fold and make a prediction for the test subset
    preds_ensemble = []
    y_stack = None
    samples = None
    for fold_nr in range(len(fold_list)):
        print("Start inference for fold with:", fold_nr, "Model type:", model_type)
        index_list, preds, labels = run_prediction(fold_nr, model_type=model_type)
        preds_ensemble.append(preds)
        if y_stack is None : y_stack = labels
        if samples is None : samples = index_list

    print(samples)

    # Preprocess prediction ensemble
    preds_ensemble = np.array(preds_ensemble)
    preds_ensemble = np.swapaxes(preds_ensemble, 0, 1)
    
    # Initialize comparison result lists
    comparison_methods = []
    comparison_preds = []

    # Run aggregate inference
    print("Start inference of aggregate functions for " + model_type)
    for agg_key in aggregate_dict:
        # Initialize aggregate function
        aggregate_function = aggregate_dict[agg_key]()
        # Make a prediction
        preds_final = []
        for i in range(preds_ensemble.shape[0]):
            pred_sample = aggregate_function.aggregate(preds_ensemble[i,:,:])
            preds_final.append(pred_sample)
        # Convert prediction list to NumPy
        preds_final = np.asarray(preds_final)

        # Store predictions to disk
        df_index = pd.DataFrame(data={"index": samples})
        df_pd = pd.DataFrame(data=preds_final, columns=["pd_" + i for i in class_names])
        df_gt = pd.DataFrame(data=y_stack, columns=["gt_" + i for i in class_names])
        df_merged = pd.concat([df_index, df_pd, df_gt], axis=1, sort=False)
        path_pred_dir = os.path.join(path_results, "predictions")
        path_preds = os.path.join(path_pred_dir, 
                                "preds.ensemble." + model_type + "." + agg_key + ".test.csv")
        df_merged.to_csv(path_preds, index=False)

        # Evaluate performance
        path_eval_dir = os.path.join(path_results, "evaluations")
        evaluate_performance(
            preds=preds_final,
            labels=y_stack,
            out_path=path_eval_dir,
            class_names=class_names,
            suffix="ensemble." + model_type + "." + agg_key + ".test",
        )
        # add to comparison list
        comparison_methods.append(agg_key)
        comparison_preds.append(preds_final)

    # Pass prediction ensemble to evaluation function
    path_eval_dir = os.path.join(path_results, "evaluations")
    evaluate_comparison(comparison_preds, y_stack, 
                        out_path=path_eval_dir, 
                        model_names=comparison_methods,
                        class_names=class_names,
                        suffix=model_type,
                        macro_average_classes=True)

#-----------------------------------------------------#
#                        Runner                       #
#-----------------------------------------------------#
""" 
    This is the runner of the script.
    
    Here, the large functions like run_bagging_training and XYZ will be called. 
"""
# Here we just iterate over each fold and train the AUCMEDI model
for fold_nr, fold in enumerate(fold_list):
    print("Start training with architecture ", architecture, "for fold with number ", fold_nr)
    run_bagging_training(fold, fold_nr)

# That's it, now we have a trained bagging model 
# Let's try it out on the testing set
for model_type in ["best_f1", "best_loss", "best_auc", "last"]:
    print(f"Start inference for Model type: {model_type}")
    run_bagging_prediction(model_type)