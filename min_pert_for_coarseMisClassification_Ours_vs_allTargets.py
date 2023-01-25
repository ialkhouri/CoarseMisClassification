import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from numpy import linalg as LA
from Functions import MinimumNorm_randomFGSM_Coarse_using_Targets, MinimumNorm_randomFGSM_Coarse_13_579_02468, MinimumNorm_randomFGSM_Coarse_013_79_24568, MinimumNorm_randomFGSM_Coarse_012_345_6789, MinimumNorm_randomFGSM_Coarse_13_48_579_026, MinimumNorm_randomFGSM_Coarse_012_34_567_89, MinimumNorm_randomFGSM_Coarse_06_148_23_579, MinimumNorm_randomFGSM
import time

#######################################################################################################################################################################
################################ fmnist  #############################################################################################################################
########################################################################################################################################################################

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# reshape data to fit model
X_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
X_test_org = test_images.reshape(test_images.shape[0], 28, 28, 1)
# normalization:
X_train, X_test_org = (X_train/255), (X_test_org/255)
#X_train, X_test = (X_train/255)-0.5, (X_test/255)-0.5
y_train_org = np_utils.to_categorical(train_labels,10)
y_test_org = np_utils.to_categorical(test_labels,10)

#######################################################################################################################################################################
################################ Trained NN #################################################################################################
#######################################################################################################################################################################
trained_model = load_model("/home/ismail/pycharmProjects/SSLTL_project/APGD-attak/FMNIST_trained_OSC_no_conv.h5")



mapping = [[0, 1, 3], [7, 9], [2, 4, 5, 6, 8]]
#mapping = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
#mapping = [[1, 3], [5, 7, 9], [0, 2, 4, 6, 8]]

#mapping = [[1, 3], [4, 8], [5, 7, 9], [0 , 2, 6]]
#mapping = [[0,1, 2], [3,4], [5, 6, 7], [8, 9]]
#mapping = [[0,6], [1,4,8], [2,3], [5,7,9]]


#test_index = 123

min_pert_list_All = []
min_pert_list_Ours = []

run_time_list_All = []
run_time_list_Ours = []

min_pert_list_any = []
run_time_list_any = []


#for test_index in range(10):
for test_index in [1]:
    # ###### Below is for the standard (ANY) mis-classification:
    # st = time.time()
    # x_adv_any, min_pert_any, _ =  MinimumNorm_randomFGSM(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
    #                                                       true_label_OHE = y_test_org[test_index].reshape(1,10),
    #                                                       trained_classifier_SM = trained_model)
    # et = time.time()
    # run_time_any = et-st
    #
    # ## save:
    # min_pert_list_any.append(min_pert_any)
    # run_time_list_any.append(run_time_any)
    #
    # print("Index = ", [test_index], " min pert Any = ", [min_pert_any],
    #       "run_time Any  ", [run_time_any])

    ##### Below is for the coarse mis-classification:
    st = time.time()
    x_adv_all, min_pert_all = MinimumNorm_randomFGSM_Coarse_using_Targets(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
                                                         true_label_OHE = y_test_org[test_index].reshape(1,10),
                                                         mapping=mapping,
                                                         trained_classifier_SM = trained_model)
    et = time.time()
    run_time_all = et-st
    ## save:
    min_pert_list_All.append(min_pert_all)
    run_time_list_All.append(run_time_all)

    st = time.time()
    x_adv_ours, min_pert_ours, _ = MinimumNorm_randomFGSM_Coarse_013_79_24568(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
                                                         true_label_OHE = y_test_org[test_index].reshape(1,10),
                                                         trained_classifier_SM = trained_model)
    et = time.time()
    run_time_ours = et-st

    ## save:
    min_pert_list_Ours.append(min_pert_ours)
    run_time_list_Ours.append(run_time_ours)

    #print("Index = ",[test_index]," AllTar vs. Ours in min pert = ", [min_pert_all, min_pert_ours], "AllTar vs. Ours run_time  ", [run_time_all,run_time_ours])


############ The final results are:

print("AllTargets: [Avg pert +/- std] = ", [np.mean(min_pert_list_All), np.std(min_pert_list_All)],   " [Avg runtime +/- std = ]", [np.mean(run_time_list_All), np.std(run_time_list_All)])
print("Ours      : [Avg pert +/- std] = ", [np.mean(min_pert_list_Ours), np.std(min_pert_list_Ours)], " [Avg runtime +/- std = ]", [np.mean(run_time_list_Ours), np.std(run_time_list_Ours)])

#print("Any min-classification: [Avg pert +/- std] = ", [np.mean(min_pert_list_any), np.std(min_pert_list_any)],   " [Avg runtime +/- std = ]", [np.mean(run_time_list_any), np.std(run_time_list_any)])