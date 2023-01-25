import tensorflow as tf
import numpy as np
from keras.utils import np_utils


######## Standard mis-classification

def BoundRestricted_randomFGSM(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry standard, this is equal to 0.3 for MNIST

    Returns
    x_adv: Eager tensor with the same shape as the input image
    -------

    """
    input = tf.Variable(input_image, dtype=tf.float32)

    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    ## 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        y_pred_with_delta = trained_classifier_SM(input + delta)
        loss_function = tf.keras.losses.MeanSquaredError()
        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_value = loss_function(true_label_OHE, y_pred_with_delta)
    grad = tape.gradient(loss_value, delta)

    ## 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    ## 4 update the perturbations:
    alpha = 1.25*pert_bound_eps
    delta_update = delta + alpha * grad_sign

    ## 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv

def MinimumNorm_randomFGSM(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause any mis-classification
    result: pass or fail
    -------

    """
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)
        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        true_label  = np.argmax(true_label_OHE)
        result = "FAIL"
        if perturbed_label != true_label:
            result = "PASS"
            #print("MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break
    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result


def BoundRestricted_randomFGSM_Targeted(input_image, target_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    target_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry standard, this is equal to 0.3 for MNIST

    Returns
    x_adv: Eager tensor with the same shape as the input image
    -------

    """
    input = tf.Variable(input_image, dtype=tf.float32)

    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    ## 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        y_pred_with_delta = trained_classifier_SM(input + delta)
        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(target_label_OHE, y_pred_with_delta)
    grad = tape.gradient(loss_value, delta)

    ## 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    ## 4 update the perturbations:
    alpha = 1.25*pert_bound_eps
    delta_update = delta - alpha * grad_sign

    ## 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv
def MinimumNorm_randomFGSM_Targeted(input_image, target_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause targeted mis-classification
    result: pass or fail
    -------

    """
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted label with pert == the target label, then, exit...
        x_adv = BoundRestricted_randomFGSM_Targeted(input_image=input_image,
                                           target_label_OHE=target_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)
        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        target_label  = np.argmax(target_label_OHE)
        result = "FAIL"
        if perturbed_label == target_label:
            result = "PASS"
            #print("targeted MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break
    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result


######## Coarse mis-classification (trying all tagrets in S'_T, adn then select the minimum)
def MinimumNorm_randomFGSM_Coarse_using_Targets(input_image, true_label_OHE,mapping, trained_classifier_SM):

    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax,
    mapping: list of lists representing the mapping of labels

    Returns
    x_adv_opt: Eager tensor with the same shape as the input image
    min_pert_opt: the minimum pert required to cause coarse mis-classification
    -------

    """

    true_label = np.argmax(true_label_OHE)
    S_T = [item for item in mapping if true_label in item][0]
    S_T_comp = [item for item in list(range(10)) if item not in S_T]

    #### start the loop:
    list_of_min_pert = []
    list_of_x_adv = []
    for target_label in S_T_comp:

        x_adv, min_l_inf_pert_norm, result = MinimumNorm_randomFGSM_Targeted(
            input_image=input_image,
            target_label_OHE=np_utils.to_categorical(target_label, 10).reshape(1, 10),
            trained_classifier_SM=trained_classifier_SM)

        if result == "PASS":
            # save the x_adv and its norm in a list or a dict
            list_of_min_pert.append(min_l_inf_pert_norm)
            list_of_x_adv.append(x_adv)

    ### Here select the minimum:
    minimum_index = np.argmin(list_of_min_pert)

    min_pert_opt = list_of_min_pert[minimum_index]
    x_adv_opt = list_of_x_adv[minimum_index]

    return x_adv_opt, min_pert_opt



######## Coarse mis-classification (our approach)

def BoundRestricted_randomFGSM_Coarse_13_579_02468(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer (or list of integers) used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry protocol, this is equal to 0.3 for MNIST

    Returns
    -------
    x_adv: Eager tensor with the same shape as the input image aiming at a coarse mis-classification
    """
    input = tf.Variable(input_image, dtype=tf.float32)
    mapping = [[1, 3], [5, 7, 9], [0, 2, 4, 6, 8]]

    true_label = np.argmax(true_label_OHE)
    true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]


    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    eps = 0.3
    alpha = 1.25 * pert_bound_eps
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    # 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        # tape.watch(input+delta)
        y_pred_with_delta = trained_classifier_SM(input + delta)

        # Let's hard code the mapping to test:
        cor_1 = tf.math.add_n([y_pred_with_delta[:, 1], y_pred_with_delta[:, 3]])
        cor_2 = tf.math.add_n([y_pred_with_delta[:, 5], y_pred_with_delta[:, 7], y_pred_with_delta[:, 9]])
        cor_3 = tf.math.add_n([y_pred_with_delta[:, 0], y_pred_with_delta[:, 2], y_pred_with_delta[:, 4], y_pred_with_delta[:, 6],y_pred_with_delta[:, 8]])

        y_pred_coarser_with_delta = tf.stack([cor_1, cor_2, cor_3], axis=1)

        # we need y_true_coarser: a numpy array of size 1 X M_c that is obtained from y_true
        y_true_coarser = np_utils.to_categorical(true_coarse_label, len(mapping))

        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(y_true_coarser.reshape(1, len(mapping)), y_pred_coarser_with_delta)

    grad = tape.gradient(loss_value, delta)

    # 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    # 4 update the perturbations:
    delta_update = delta + alpha * grad_sign

    # 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv
def BoundRestricted_randomFGSM_Coarse_013_79_24568(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer (or list of integers) used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry protocol, this is equal to 0.3 for MNIST

    Returns
    -------
    x_adv: Eager tensor with the same shape as the input image aiming at a coarse mis-classification
    """
    input = tf.Variable(input_image, dtype=tf.float32)
    mapping = [[0, 1, 3], [7, 9], [2, 4, 5, 6, 8]]

    true_label = np.argmax(true_label_OHE)
    true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]


    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    eps = 0.3
    alpha = 1.25 * pert_bound_eps
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    # 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        # tape.watch(input+delta)
        y_pred_with_delta = trained_classifier_SM(input + delta)

        # Let's hard code the mapping to test:
        cor_1 = tf.math.add_n([y_pred_with_delta[:, 0], y_pred_with_delta[:, 1], y_pred_with_delta[:, 3]])
        cor_2 = tf.math.add_n([y_pred_with_delta[:, 7], y_pred_with_delta[:, 9]])
        cor_3 = tf.math.add_n([y_pred_with_delta[:, 2], y_pred_with_delta[:, 4], y_pred_with_delta[:, 5], y_pred_with_delta[:, 6],y_pred_with_delta[:, 8]])

        y_pred_coarser_with_delta = tf.stack([cor_1, cor_2, cor_3], axis=1)

        # we need y_true_coarser: a numpy array of size 1 X M_c that is obtained from y_true
        y_true_coarser = np_utils.to_categorical(true_coarse_label, len(mapping))

        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(y_true_coarser.reshape(1, len(mapping)), y_pred_coarser_with_delta)

    grad = tape.gradient(loss_value, delta)

    # 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    # 4 update the perturbations:
    delta_update = delta + alpha * grad_sign

    # 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv
def BoundRestricted_randomFGSM_Coarse_012_345_6789(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer (or list of integers) used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry protocol, this is equal to 0.3 for MNIST

    Returns
    -------
    x_adv: Eager tensor with the same shape as the input image aiming at a coarse mis-classification
    """
    input = tf.Variable(input_image, dtype=tf.float32)
    mapping = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

    true_label = np.argmax(true_label_OHE)
    true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]


    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    eps = 0.3
    alpha = 1.25 * pert_bound_eps
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    # 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        # tape.watch(input+delta)
        y_pred_with_delta = trained_classifier_SM(input + delta)

        # Let's hard code the mapping to test:
        cor_1 = tf.math.add_n([y_pred_with_delta[:, 0], y_pred_with_delta[:, 1], y_pred_with_delta[:, 2]])
        cor_2 = tf.math.add_n([y_pred_with_delta[:, 3], y_pred_with_delta[:, 4], y_pred_with_delta[:, 5]])
        cor_3 = tf.math.add_n([y_pred_with_delta[:, 6], y_pred_with_delta[:, 7], y_pred_with_delta[:, 8], y_pred_with_delta[:, 9]])

        y_pred_coarser_with_delta = tf.stack([cor_1, cor_2, cor_3], axis=1)

        # we need y_true_coarser: a numpy array of size 1 X M_c that is obtained from y_true
        y_true_coarser = np_utils.to_categorical(true_coarse_label, len(mapping))

        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(y_true_coarser.reshape(1, len(mapping)), y_pred_coarser_with_delta)

    grad = tape.gradient(loss_value, delta)

    # 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    # 4 update the perturbations:
    delta_update = delta + alpha * grad_sign

    # 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv


def BoundRestricted_randomFGSM_Coarse_13_48_579_026(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer (or list of integers) used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry protocol, this is equal to 0.3 for MNIST

    Returns
    -------
    x_adv: Eager tensor with the same shape as the input image aiming at a coarse mis-classification
    """
    input = tf.Variable(input_image, dtype=tf.float32)
    mapping = [[1,3], [4,8], [5,7,9], [0,2,6]]

    true_label = np.argmax(true_label_OHE)
    true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]


    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    eps = 0.3
    alpha = 1.25 * pert_bound_eps
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    # 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        # tape.watch(input+delta)
        y_pred_with_delta = trained_classifier_SM(input + delta)

        # Let's hard code the mapping to test:
        cor_1 = tf.math.add_n([y_pred_with_delta[:, 1], y_pred_with_delta[:, 3]])
        cor_2 = tf.math.add_n([y_pred_with_delta[:, 4], y_pred_with_delta[:, 8]])
        cor_3 = tf.math.add_n([y_pred_with_delta[:, 5], y_pred_with_delta[:, 7], y_pred_with_delta[:, 9]])
        cor_4 = tf.math.add_n([y_pred_with_delta[:, 0], y_pred_with_delta[:, 2], y_pred_with_delta[:, 6]])

        y_pred_coarser_with_delta = tf.stack([cor_1, cor_2, cor_3, cor_4], axis=1)

        # we need y_true_coarser: a numpy array of size 1 X M_c that is obtained from y_true
        y_true_coarser = np_utils.to_categorical(true_coarse_label, len(mapping))

        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(y_true_coarser.reshape(1, len(mapping)), y_pred_coarser_with_delta)

    grad = tape.gradient(loss_value, delta)

    # 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    # 4 update the perturbations:
    delta_update = delta + alpha * grad_sign

    # 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv
def BoundRestricted_randomFGSM_Coarse_012_34_567_89(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer (or list of integers) used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry protocol, this is equal to 0.3 for MNIST

    Returns
    -------
    x_adv: Eager tensor with the same shape as the input image aiming at a coarse mis-classification
    """
    input = tf.Variable(input_image, dtype=tf.float32)
    mapping = [[0,1,2], [3,4], [5,6,7], [8,9]]

    true_label = np.argmax(true_label_OHE)
    true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]


    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    eps = 0.3
    alpha = 1.25 * pert_bound_eps
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    # 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        # tape.watch(input+delta)
        y_pred_with_delta = trained_classifier_SM(input + delta)

        # Let's hard code the mapping to test:
        cor_1 = tf.math.add_n([y_pred_with_delta[:, 0], y_pred_with_delta[:, 1], y_pred_with_delta[:, 2]])
        cor_2 = tf.math.add_n([y_pred_with_delta[:, 3], y_pred_with_delta[:, 4]])
        cor_3 = tf.math.add_n([y_pred_with_delta[:, 5], y_pred_with_delta[:, 6], y_pred_with_delta[:, 7]])
        cor_4 = tf.math.add_n([y_pred_with_delta[:, 8], y_pred_with_delta[:, 9]])

        y_pred_coarser_with_delta = tf.stack([cor_1, cor_2, cor_3, cor_4], axis=1)

        # we need y_true_coarser: a numpy array of size 1 X M_c that is obtained from y_true
        y_true_coarser = np_utils.to_categorical(true_coarse_label, len(mapping))

        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(y_true_coarser.reshape(1, len(mapping)), y_pred_coarser_with_delta)

    grad = tape.gradient(loss_value, delta)

    # 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    # 4 update the perturbations:
    delta_update = delta + alpha * grad_sign

    # 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv
def BoundRestricted_randomFGSM_Coarse_06_148_23_579(input_image, true_label_OHE, seed, trained_classifier_SM, pert_bound_eps):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    seed: an integer (or list of integers) used to initialize the perturbations randomly
    trained_classifier_SM: Keras model WITH softmax
    pert_bound_eps: a scaler. For FMNIST AT, using Madry protocol, this is equal to 0.3 for MNIST

    Returns
    -------
    x_adv: Eager tensor with the same shape as the input image aiming at a coarse mis-classification
    """
    input = tf.Variable(input_image, dtype=tf.float32)
    mapping = [[0,6], [1,4,8], [2,3], [5,7,9]]

    true_label = np.argmax(true_label_OHE)
    true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]


    ## 1 Init delta = Uniform(-eps,eps)
    np.random.seed(seed=seed)
    delta_numpy = np.random.uniform(low=-pert_bound_eps, high=pert_bound_eps, size=(1, 28, 28, 1))
    eps = 0.3
    alpha = 1.25 * pert_bound_eps
    # convert to tf tensor
    delta = tf.Variable(delta_numpy, dtype=tf.float32)

    # 2 obtain the gradient of the CCE (not logits) of the prediction of x + delta w.r.t. delta:
    with tf.GradientTape() as tape:
        # Forward Pass
        # tape.watch(input+delta)
        y_pred_with_delta = trained_classifier_SM(input + delta)

        # Let's hard code the mapping to test:
        cor_1 = tf.math.add_n([y_pred_with_delta[:, 0], y_pred_with_delta[:, 6]])
        cor_2 = tf.math.add_n([y_pred_with_delta[:, 1], y_pred_with_delta[:, 4], y_pred_with_delta[:, 8]])
        cor_3 = tf.math.add_n([y_pred_with_delta[:, 2], y_pred_with_delta[:, 3]])
        cor_4 = tf.math.add_n([y_pred_with_delta[:, 5], y_pred_with_delta[:, 7], y_pred_with_delta[:, 9]])

        y_pred_coarser_with_delta = tf.stack([cor_1, cor_2, cor_3, cor_4], axis=1)

        # we need y_true_coarser: a numpy array of size 1 X M_c that is obtained from y_true
        y_true_coarser = np_utils.to_categorical(true_coarse_label, len(mapping))

        #loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss_function = tf.keras.losses.MeanSquaredError()
        loss_value = loss_function(y_true_coarser.reshape(1, len(mapping)), y_pred_coarser_with_delta)

    grad = tape.gradient(loss_value, delta)

    # 3 obtain the sign of the grad
    grad_sign = tf.math.sign(grad)

    # 4 update the perturbations:
    delta_update = delta + alpha * grad_sign

    # 5 clip w.r.t. epsilon:
    delta_clipped = tf.clip_by_value(delta_update, clip_value_min=-pert_bound_eps, clip_value_max=pert_bound_eps)

    # 6 obtain the perturbed tensor
    x_adv = input + delta_clipped

    return x_adv


def MinimumNorm_randomFGSM_Coarse_13_579_02468(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause coarse mis-classification
    result: pass or fail
    -------

    """
    mapping = [[1, 3], [5, 7, 9], [0, 2, 4, 6, 8]]
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM_Coarse_13_579_02468(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)

        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        predicted_coarse = [ii for ii, lst in enumerate(mapping) if perturbed_label in lst][0]
        true_label = np.argmax(true_label_OHE)
        true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]

        result = "FAIL"
        if predicted_coarse != true_coarse_label:
            result = "PASS"
            print("Coarse MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break

    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result
def MinimumNorm_randomFGSM_Coarse_013_79_24568(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause coarse mis-classification
    result: pass or fail
    -------

    """
    mapping = [[0, 1, 3], [7, 9], [2, 4, 5, 6, 8]]
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM_Coarse_013_79_24568(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)

        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        predicted_coarse = [ii for ii, lst in enumerate(mapping) if perturbed_label in lst][0]
        true_label = np.argmax(true_label_OHE)
        true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]

        result = "FAIL"
        if predicted_coarse != true_coarse_label:
            result = "PASS"
            print("Coarse MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break

    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result
def MinimumNorm_randomFGSM_Coarse_012_345_6789(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause coarse mis-classification
    result: pass or fail
    -------

    """
    mapping = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM_Coarse_012_345_6789(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)

        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        predicted_coarse = [ii for ii, lst in enumerate(mapping) if perturbed_label in lst][0]
        true_label = np.argmax(true_label_OHE)
        true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]

        result = "FAIL"
        if predicted_coarse != true_coarse_label:
            result = "PASS"
            print("Coarse MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break

    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result


def MinimumNorm_randomFGSM_Coarse_13_48_579_026(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause coarse mis-classification
    result: pass or fail
    -------

    """
    mapping = [[1, 3], [4, 8], [5, 7, 9], [0, 2, 6]]
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM_Coarse_13_48_579_026(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)

        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        predicted_coarse = [ii for ii, lst in enumerate(mapping) if perturbed_label in lst][0]
        true_label = np.argmax(true_label_OHE)
        true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]

        result = "FAIL"
        if predicted_coarse != true_coarse_label:
            result = "PASS"
            print("Coarse MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break

    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result
def MinimumNorm_randomFGSM_Coarse_012_34_567_89(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause coarse mis-classification
    result: pass or fail
    -------

    """
    mapping = [[0,1,2], [3,4], [5,6,7], [8,9]]
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM_Coarse_012_34_567_89(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)

        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        predicted_coarse = [ii for ii, lst in enumerate(mapping) if perturbed_label in lst][0]
        true_label = np.argmax(true_label_OHE)
        true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]

        result = "FAIL"
        if predicted_coarse != true_coarse_label:
            result = "PASS"
            print("Coarse MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break

    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result
def MinimumNorm_randomFGSM_Coarse_06_148_23_579(input_image, true_label_OHE, trained_classifier_SM):
    """

    Parameters
    ----------
    input_image: numpy array of size (1, rows, columns, channels). For example, (1,28,28,1) for MNIST
    true_label_OHE: numpy array of size (1, number _of_classes). For example, (1,10) for MNIST
    trained_classifier_SM: Keras model WITH softmax

    Returns
    x_adv: Eager tensor with the same shape as the input image
    min_l_inf_pert_norm: the minimum pert required to cause coarse mis-classification
    result: pass or fail
    -------

    """
    mapping = [[0,6], [1,4,8], [2,3], [5,7,9]]
    eps_range = list(np.arange(0.01, 1.0, 0.01))

    for bound_eps in eps_range:
        # apply the attack, if the predicted labels \neq the predicted label with pert, then, exit...
        x_adv = BoundRestricted_randomFGSM_Coarse_06_148_23_579(input_image=input_image,
                                           true_label_OHE=true_label_OHE,
                                           seed=eps_range.index(bound_eps) + 1,
                                           trained_classifier_SM=trained_classifier_SM,
                                           pert_bound_eps=bound_eps)

        perturbed_output = trained_classifier_SM(x_adv)
        perturbed_label = np.argmax(perturbed_output.numpy())
        predicted_coarse = [ii for ii, lst in enumerate(mapping) if perturbed_label in lst][0]
        true_label = np.argmax(true_label_OHE)
        true_coarse_label = [ii for ii, lst in enumerate(mapping) if true_label in lst][0]

        result = "FAIL"
        if predicted_coarse != true_coarse_label:
            result = "PASS"
            print("Coarse MinimumNorm_randomFGSM is pass at eps = ", bound_eps)
            break

    min_l_inf_pert_norm = bound_eps
    return x_adv, min_l_inf_pert_norm, result

