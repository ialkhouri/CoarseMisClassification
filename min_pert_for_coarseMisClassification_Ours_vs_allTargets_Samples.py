import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from numpy import linalg as LA
from Functions import MinimumNorm_randomFGSM_Coarse_using_Targets, MinimumNorm_randomFGSM_Coarse_13_579_02468, MinimumNorm_randomFGSM_Coarse_013_79_24568, MinimumNorm_randomFGSM_Coarse_012_345_6789, MinimumNorm_randomFGSM_Coarse_13_48_579_026, MinimumNorm_randomFGSM_Coarse_012_34_567_89, MinimumNorm_randomFGSM_Coarse_06_148_23_579, MinimumNorm_randomFGSM
import matplotlib.pyplot as plt

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



#mapping = [[0, 1, 3], [7, 9], [2, 4, 5, 6, 8]]
mapping = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
#mapping = [[1, 3], [5, 7, 9], [0, 2, 4, 6, 8]]

#mapping = [[1, 3], [4, 8], [5, 7, 9], [0 , 2, 6]]
#mapping = [[0,1, 2], [3,4], [5, 6, 7], [8, 9]]
#mapping = [[0,6], [1,4,8], [2,3], [5,7,9]]


#test_index = 123

for test_index in [20, 11, 28]:
#for test_index in range(50):
    x_adv_all, min_pert_all = MinimumNorm_randomFGSM_Coarse_using_Targets(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
                                                         true_label_OHE = y_test_org[test_index].reshape(1,10),
                                                         mapping=mapping,
                                                         trained_classifier_SM = trained_model)

    perturbed_output_all = trained_model(x_adv_all)
    perturbed_label_all = np.argmax(perturbed_output_all.numpy())
    predicted_coarse_all = [ii for ii, lst in enumerate(mapping) if perturbed_label_all in lst][0]
    true_label_all = np.argmax(y_test_org[test_index].reshape(1,10))
    true_coarse_label_all = [ii for ii, lst in enumerate(mapping) if true_label_all in lst][0]

    x_adv_ours, min_pert_ours, _ = MinimumNorm_randomFGSM_Coarse_012_345_6789(input_image = X_test_org[test_index].reshape(1, 28, 28, 1),
                                                         true_label_OHE = y_test_org[test_index].reshape(1,10),
                                                         trained_classifier_SM = trained_model)

    perturbed_output_ours = trained_model(x_adv_ours)
    perturbed_label_ours = np.argmax(perturbed_output_all.numpy())
    predicted_coarse_ours = [ii for ii, lst in enumerate(mapping) if perturbed_label_ours in lst][0]
    true_label_ours = np.argmax(y_test_org[test_index].reshape(1, 10))
    true_coarse_label_ours = [ii for ii, lst in enumerate(mapping) if true_label_ours in lst][0]

    print("test index = ", [test_index], "true coarse = ", true_coarse_label_ours, "pert. coarse [all vs. ours] = ", [predicted_coarse_all, predicted_coarse_ours], "pert bound [all vs. ours] = ", [min_pert_all, min_pert_ours])



    image = plt.imshow(X_test_org[test_index].reshape(28,28))
    plt.show()
    image = plt.imshow(x_adv_all.numpy().reshape(28,28))
    plt.show()
    image = plt.imshow(x_adv_ours.numpy().reshape(28,28))
    plt.show()
