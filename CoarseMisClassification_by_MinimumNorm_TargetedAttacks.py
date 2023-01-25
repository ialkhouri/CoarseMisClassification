import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from numpy import linalg as LA
from Functions import MinimumNorm_randomFGSM_Targeted, MinimumNorm_randomFGSM_Coarse_using_Targets


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

# ########### Testing the function:
# test_index = 8888
# #mapping = [[1, 3], [5, 7, 9], [0, 2, 4, 6, 8]]
# mapping = [[0,1,2], [3,4], [5,6,7], [8,9]]
# ### set S_T and S_T_comp ==>
# #true_label = np.argmax(true_label_OHE)
# true_label = test_labels[test_index]
# S_T = [item for item in mapping if true_label in item][0]
# S_T_comp = [item for item in list(range(10)) if item not in S_T]
#
# #### start the loop:
#
# list_of_min_pert = []
# list_of_x_adv = []
# for target_label in S_T_comp:
#
#     x_adv, min_l_inf_pert_norm, result = MinimumNorm_randomFGSM_Targeted(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
#                                                     target_label_OHE = np_utils.to_categorical(target_label,10).reshape(1,10),
#                                                     trained_classifier_SM = trained_model)
#
#     if result == "PASS":
#         # save the x_adv and its norm in a list or a dict
#         list_of_min_pert.append(min_l_inf_pert_norm)
#         list_of_x_adv.append(x_adv)
#
#
# ### Here select the minimum:
# print("break")
#
# minimum_index = np.argmin(list_of_min_pert)
#
# min_pert_opt = list_of_min_pert[minimum_index]
# x_adv_opt = list_of_x_adv[minimum_index]
#
# print("The minimum is ", min_pert_opt)



########### Testing the function:
test_index = 5555
#mapping = [[1, 3], [5, 7, 9], [0, 2, 4, 6, 8]]
mapping = [[0,1,2], [3,4], [5,6,7], [8,9]]

x_adv, min_pert = MinimumNorm_randomFGSM_Coarse_using_Targets(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
                                                     true_label_OHE = y_test_org[test_index].reshape(1,10),
                                                     mapping=mapping,
                                                     trained_classifier_SM = trained_model)

print("The minimum is ", min_pert)