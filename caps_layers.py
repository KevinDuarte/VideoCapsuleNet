import tensorflow as tf
import numpy as np
from functools import reduce
import config


epsilon = 1e-7
mdim = config.pose_dimension
mdim2 = mdim*mdim


def create_prim_conv3d_caps(inputs, channels, kernel_size, strides, name, padding='VALID', activation=None):
    """
    Creates the primary capsule layer from a set of input feature maps with shape (N, D_in, H_in, W_in, C_in)

    :param inputs: Input feature maps
    :param channels: Number of capsule types
    :param kernel_size: Size of kernel which is applied to the feature maps. Should be 3 dimensional (K_t, K_h, K_w).
    :param strides: The striding which the kernel uses. Should be 3 dimensional (S_t, S_h, S_w)
    :param name: Name given to the layer
    :param padding: Whether or not padding should be applied. Should be either 'VALID' or 'SAME', for no padding and
    padding respectively.
    :param activation: The activation function used for the pose matrices.
    :return: Returns capsules of the form (poses, activations). Poses have shape (N, D_out, H_out, W_out, C_out, M)
    where M is the height*width of the pose matrix. Activations have shape (N, D_out, H_out, W_out, C_out, 1).
    """
    batch_size = tf.shape(inputs)[0]
    poses = tf.layers.conv3d(inputs=inputs, filters=channels * mdim2, kernel_size=kernel_size,
                             strides=strides, padding=padding, activation=activation, name=name+'_pose')

    _, d, h, w, _ = poses.get_shape().as_list()
    d, h, w = map(int, [d, h, w])

    pose = tf.reshape(poses, (batch_size, d, h, w, channels, mdim2))

    acts = tf.layers.conv3d(inputs=inputs, filters=channels, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=tf.nn.sigmoid, name=name+'_act')
    activation = tf.reshape(acts, (batch_size, d, h, w, channels, 1))

    return pose, activation


def em_routing(v, a_i, beta_v, beta_a, n_iterations=3):
    """
    Performs the EM-routing (https://openreview.net/pdf?id=HJWLfGWRb).

    Note:One change from the original algorithm is used to ensure numerical stability. The cost used to calculate the
    activations are normalized, which makes the output activations relative to each other (i.e. an activation is high if
    its cost is lower than the other capsules' costs). This works for most applications, but it leads to some issues if
    your application necessitates all, or most, capsules to be active.


    :param v: The votes for the higher level capsules. Shape - (N, C_i, C_j, M)
    :param a_i: The activations of the lower level capsules. Shape - (N, C_i, 1)
    :param beta_v: The beta_v parameter for routing (check original EM-routing paper for details)
    :param beta_a: The beta_a parameter for routing (check original EM-routing paper for details)
    :param n_iterations: Number of iterations which routing takes place.
    :return: Returns capsules of the form (poses, activations). Poses have shape (N, C_out, M) where M is the
    height*width of the pose matrix. Activations have shape (N, C_out, 1).
    """
    batch_size = tf.shape(v)[0]
    _, _, n_caps_j, mat_len = v.get_shape().as_list()
    n_caps_j, mat_len = map(int, [n_caps_j, mat_len])
    n_caps_i = tf.shape(v)[1]

    a_i = tf.expand_dims(a_i, axis=-1)

    # Prior probabilities for routing
    r = tf.ones(shape=(batch_size, n_caps_i, n_caps_j, 1), dtype=tf.float32)/float(n_caps_j)
    r = tf.multiply(r, a_i)

    den = tf.reduce_sum(r, axis=1, keep_dims=True) + epsilon

    # Mean: shape=(N, 1, Ch_j, mat_len)
    m_num = tf.reduce_sum(v*r, axis=1, keep_dims=True)
    m = m_num/(den + epsilon)

    # Stddev: shape=(N, 1, Ch_j, mat_len)
    s_num = tf.reduce_sum(r * tf.square(v - m), axis=1, keep_dims=True)
    s = s_num/(den + epsilon)

    # cost_h: shape=(N, 1, Ch_j, mat_len)
    cost = (beta_v + tf.log(tf.sqrt(s + epsilon) + epsilon)) * den
    # cost_h: shape=(N, 1, Ch_j, 1)
    cost = tf.reduce_sum(cost, axis=-1, keep_dims=True)

    # calculates the mean and std_deviation of the cost
    cost_mean = tf.reduce_mean(cost, axis=-2, keep_dims=True)
    cost_stdv = tf.sqrt(
        tf.reduce_sum(
            tf.square(cost - cost_mean), axis=-2, keep_dims=True
        ) / n_caps_j + epsilon
    )

    # calculates the activations for the capsules in layer j
    a_j = tf.sigmoid(float(config.inv_temp) * (beta_a + (cost_mean - cost) / (cost_stdv + epsilon)))
    # a_j = tf.sigmoid(float(config.inv_temp) * (beta_a - cost)) # may lead to numerical instability

    def condition(mean, stdsqr, act_j, counter):
        return tf.less(counter, n_iterations)

    def route(mean, stdsqr, act_j, counter):
        exp = tf.reduce_sum(tf.square(v - mean) / (2 * stdsqr + epsilon), axis=-1)
        coef = 0 - .5 * tf.reduce_sum(tf.log(2 * np.pi * stdsqr + epsilon), axis=-1)
        log_p_j = coef - exp

        log_ap = tf.reshape(tf.log(act_j + epsilon), (batch_size, 1, n_caps_j)) + log_p_j
        r_ij = tf.nn.softmax(log_ap + epsilon)  # ap / (tf.reduce_sum(ap, axis=-1, keep_dims=True) + epsilon)

        r_ij = tf.multiply(tf.expand_dims(r_ij, axis=-1), a_i)

        denom = tf.reduce_sum(r_ij, axis=1, keep_dims=True)
        m_numer = tf.reduce_sum(v * r_ij, axis=1, keep_dims=True)
        mean = m_numer / (denom + epsilon)

        s_numer = tf.reduce_sum(r_ij * tf.square(v - mean), axis=1, keep_dims=True)
        stdsqr = s_numer / (denom + epsilon)

        cost_h = (beta_v + tf.log(tf.sqrt(stdsqr) + epsilon)) * denom
        cost_h = tf.reduce_sum(cost_h, axis=-1, keep_dims=True)

        # these are calculated for numerical stability.
        cost_h_mean = tf.reduce_mean(cost_h, axis=-2, keep_dims=True)
        cost_h_stdv = tf.sqrt(
            tf.reduce_sum(
                tf.square(cost_h - cost_h_mean), axis=-2, keep_dims=True
            ) / n_caps_j
        )

        inv_temp = config.inv_temp + counter * config.inv_temp_delta
        act_j = tf.sigmoid(inv_temp * (beta_a + (cost_h_mean - cost_h) / (cost_h_stdv + epsilon)))
        # act_j = tf.sigmoid(inv_temp * (beta_a - cost_h)) # may lead to numerical instability

        return mean, stdsqr, act_j, tf.add(counter, 1)

    [mean, _, act_j, _] = tf.while_loop(condition, route, [m, s, a_j, 1.0])

    return tf.reshape(mean, (batch_size, n_caps_j, mat_len)), tf.reshape(act_j, (batch_size, n_caps_j, 1))


def create_coords_mat(pose, rel_center):
    """
    Create matrices for coordinate addition. The output of this function should be added to the vote matrix.

    :param pose: The incoming map of pose matrices, shape (N, ..., Ch_i, 16) where ... is the dimensions of the map can
    be 1, 2 or 3 dimensional.
    :param rel_center: whether or not the coordinates are relative to the center of the map
    :return: Returns the coordinates (padded to 16) for the incoming capsules.
    """
    batch_size = tf.shape(pose)[0]
    shape_list = [int(x) for x in pose.get_shape().as_list()[1:-2]]
    ch = int(pose.get_shape().as_list()[-2])
    n_dims = len(shape_list)

    if n_dims == 3:
        d, h, w = shape_list
    elif n_dims == 2:
        d = 1
        h, w = shape_list
    else:
        d, h = 1, 1
        w = shape_list[0]

    subs = [0, 0, 0]
    if rel_center:
        subs = [int(d / 2), int(h / 2), int(w / 2)]

    c_mats = []
    if n_dims >= 3:
        c_mats.append(tf.tile(tf.reshape(tf.range(d, dtype=tf.float32), (1, d, 1, 1, 1, 1)), [batch_size, 1, h, w, ch, 1])-subs[0])
    if n_dims >= 2:
        c_mats.append(tf.tile(tf.reshape(tf.range(h, dtype=tf.float32), (1, 1, h, 1, 1, 1)), [batch_size, d, 1, w, ch, 1])-subs[1])
    if n_dims >= 1:
        c_mats.append(tf.tile(tf.reshape(tf.range(w, dtype=tf.float32), (1, 1, 1, w, 1, 1)), [batch_size, d, h, 1, ch, 1])-subs[2])
    add_coords = tf.concat(c_mats, axis=-1)
    add_coords = tf.cast(tf.reshape(add_coords, (batch_size, d*h*w, ch, n_dims)), dtype=tf.float32)

    zeros = tf.zeros((batch_size, d*h*w, ch, mdim2-n_dims))

    return tf.concat([zeros, add_coords], axis=-1)


def get_subset(u_i, coords, activation, k):
    """
    Gets a subset of k capsules of each capsule type, based on their activation. When k=1, this is equivalent to
    "max-pooling" where the most active capsule for each capsule type is used.


    :param u_i: The incoming pose matrices shape (N, K, Ch_i, M)
    :param coords: The coords for these pose matrices (N, K, Ch_i, M)
    :param activation: The activations of the capsules (N, K, Ch_i, 1)
    :param k: Number of capsules which will be routed
    :return: New u_i, coords, and activation which only have k of the most active capsules per channel
    """
    batch_size, n_capsch_i, ch = tf.shape(u_i)[0], int(u_i.get_shape().as_list()[1]), tf.shape(u_i)[2]

    inputs_res = tf.reshape(tf.concat([u_i, coords, activation], axis=-1), (batch_size, n_capsch_i, ch, mdim2*2+1))

    trans = tf.transpose(inputs_res, [0, 2, 1, 3])

    norms = tf.reshape(trans[:, :, :, -1], (batch_size, ch, n_capsch_i))

    inds = tf.nn.top_k(norms, k).indices

    bt = tf.reshape(tf.range(batch_size), (batch_size, 1))
    bt = tf.reshape(tf.tile(bt, [1, ch * k]), (batch_size, ch * k, 1))

    ct = tf.reshape(tf.range(ch), (ch, 1))
    ct = tf.reshape(tf.tile(ct, [1, k]), (ch, k, 1))
    ct = tf.reshape(tf.tile(ct, [batch_size, 1, 1]), (batch_size, ch * k, 1))

    conc = tf.concat([bt, ct], axis=2)
    t = tf.reshape(conc, (batch_size, ch, k, 2))

    inds = tf.reshape(inds, (batch_size, ch, k, 1))
    coords = tf.concat([t, inds], axis=3)

    top_caps = tf.gather_nd(trans, coords)

    top_caps = tf.transpose(top_caps, (0, 2, 1, 3))
    top_poses = top_caps[:, :, :, :mdim2]
    top_coords = top_caps[:, :, :, mdim2:-1]
    top_acts = top_caps[:, :, :, -1:]

    return top_poses, top_coords, top_acts


def create_dense_caps(inputs, n_caps_j, name, subset_routing=-1, route_min=0.0, coord_add=False, rel_center=False,
                      ch_same_w=True):
    """
    Creates a set of capsules from a lower level capsule layer.

    :param inputs: The input capsule layer. Shape ((N, K, Ch_i, M), (N, K, Ch_i, 1)) or
    ((N, ..., Ch_i, M), (N, ..., Ch_i, 1)) where K is the number of capsules per channel and '...' is if you are
    inputting an map of capsules like W or HxW or DxHxW.
    :param n_caps_j: The number of capsules in the following layer
    :param name: Name of the capsule layer
    :param subset_routing: Route only the K most active capsules of each capsule type in the previous layer. If set to
    -1 then all capsules are routed
    :param route_min: A threshold activation to route. Only capsules above this threshold are used.
    :param coord_add: A boolean, whether or not to to do coordinate addition
    :param rel_center: A boolean, whether or not the coordinate addition is relative to the center
    :param ch_same_w: If True, then capsules of the same type use the same weights. If false, all capsules use different
    weights.
    :return: Returns a layer of capsules. Shape ((N, n_caps_j, M), (N, n_caps_j, 1))
    """
    pose, activation = inputs
    batch_size = tf.shape(pose)[0]
    shape_list = [int(x) for x in pose.get_shape().as_list()[1:]]
    ch = int(shape_list[-2])
    n_capsch_i = 1 if len(shape_list) == 2 else reduce((lambda x, y: x * y), shape_list[:-2])

    u_i = tf.reshape(pose, (batch_size, n_capsch_i, ch, mdim2))
    activation = tf.reshape(activation, (batch_size, n_capsch_i, ch, 1))
    coords = create_coords_mat(pose, rel_center) if coord_add else tf.zeros_like(u_i)

    # extracts a subset of capsules to be routed
    if subset_routing != -1:
        u_i, coords, activation = get_subset(u_i, coords, activation, k=subset_routing)
        n_capsch_i = subset_routing

    # reshapes the input capsules
    u_i = tf.reshape(u_i, (batch_size, n_capsch_i, ch, mdim, mdim))
    u_i = tf.expand_dims(u_i, axis=-3)
    u_i = tf.tile(u_i, [1, 1, 1, n_caps_j, 1, 1])

    # calculates votes
    if ch_same_w:
        weights = tf.get_variable(name=name + '_weights', shape=(ch, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))

        votes = tf.einsum('ijab,ntijbc->ntijac', weights, u_i)
        votes = tf.reshape(votes, (batch_size, n_capsch_i * ch, n_caps_j, mdim2))
    else:
        weights = tf.get_variable(name=name + '_weights', shape=(n_capsch_i, ch, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))
        votes = tf.einsum('tijab,ntijbc->ntijac', weights, u_i)
        votes = tf.reshape(votes, (batch_size, n_capsch_i * ch, n_caps_j, mdim2))

    # performs coordinate addition
    if coord_add:
        coords = tf.reshape(coords, (batch_size, n_capsch_i * ch, 1, mdim2))
        votes = votes + tf.tile(coords, [1, 1, n_caps_j, 1])

    # performs thresholding, so only capsules with activations over the threshold "route_min" are routed
    acts = tf.reshape(activation, (batch_size, n_capsch_i * ch, 1))
    activations = tf.where(tf.greater_equal(acts, tf.constant(route_min)), acts, tf.zeros_like(acts))

    # creates the routing parameters
    beta_v = tf.get_variable(name=name + '_beta_v', shape=(n_caps_j, mdim2),
                             initializer=tf.initializers.random_normal(stddev=0.1),
                             regularizer=tf.contrib.layers.l2_regularizer(0.1))
    beta_a = tf.get_variable(name=name + '_beta_a', shape=(n_caps_j, 1),
                             initializer=tf.initializers.random_normal(stddev=0.1),
                             regularizer=tf.contrib.layers.l2_regularizer(0.1))

    # performs EM-routing and returns the new capsules
    return em_routing(votes, activations, beta_v, beta_a)


def create_conv3d_caps(inputs, channels, kernel_size, strides, name, padding='VALID', subset_routing=-1, route_min=0.0,
                       coord_add=False, rel_center=True, route_mean=True, ch_same_w=True):
    """
    Creates capsules from a lower level capsule layer using 3d-convolutional capsule routing.

    :param inputs: The input capsule layer. Shape ((N, D_in, H_in, W_in, Ch_i, M), (N, D_in, H_in, W_in, Ch_i, 1))
    :param channels:  The number of capsule types in the following layer
    :param kernel_size: Size of the receptive field used for routing. Should be 3 dimensional (K_t, K_h, K_w).
    :param strides: The striding which the receptive field uses. Should be 3 dimensional (S_t, S_h, S_w)
    :param name: Name of the capsule layer
    :param padding: Whether or not padding should be applied. Should be either 'VALID' or 'SAME', for no padding and
    padding respectively.
    :param subset_routing: Route only the K most active capsules of each capsule type in the previous layer. If set to
    -1 then all capsules are routed
    :param route_min: A threshold activation to route. Only capsules above this threshold are used.
    :param coord_add: A boolean, whether or not to to do coordinate addition
    :param rel_center: A boolean, whether or not the coordinate addition is relative to the center
    :param route_mean: If True, then the mean of the receptive field will be routing (capsule-pooling).
    :param ch_same_w: If True, then capsules of the same type use the same weights. If false, all capsules use different
    weights.
    :return: Returns a layer of capsules.
    Shape ((N, D_out, H_out, W_out, n_caps_j, M), (N, D_out, H_out, W_out, n_caps_j, 1))
    """
    inputs = tf.concat(inputs, axis=-1)

    # pads the input
    if padding == 'SAME':
        d_padding, h_padding, w_padding = int(float(kernel_size[0]) / 2), int(float(kernel_size[1]) / 2), int(float(kernel_size[2]) / 2)
        u_padded = tf.pad(inputs, [[0, 0], [d_padding, d_padding], [h_padding, h_padding], [w_padding, w_padding], [0, 0], [0, 0]])
    else:
        u_padded = inputs

    batch_size = tf.shape(u_padded)[0]
    _, d, h, w, ch, _ = u_padded.get_shape().as_list()
    d, h, w, ch = map(int, [d, h, w, ch])

    # gets indices for kernels
    d_offsets = [[(d_ + k) for k in range(kernel_size[0])] for d_ in range(0, d + 1 - kernel_size[0], strides[0])]
    h_offsets = [[(h_ + k) for k in range(kernel_size[1])] for h_ in range(0, h + 1 - kernel_size[1], strides[1])]
    w_offsets = [[(w_ + k) for k in range(kernel_size[2])] for w_ in range(0, w + 1 - kernel_size[2], strides[2])]

    # output dimensions
    d_out, h_out, w_out = len(d_offsets), len(h_offsets), len(w_offsets)

    # gathers the capsules into shape (N, D2, H2, W2, KD, KH, KW, Ch_in, 17)
    d_gathered = tf.gather(u_padded, d_offsets, axis=1)
    h_gathered = tf.gather(d_gathered, h_offsets, axis=3)
    w_gathered = tf.gather(h_gathered, w_offsets, axis=5)
    w_gathered = tf.transpose(w_gathered, [0, 1, 3, 5, 2, 4, 6, 7, 8])

    # obtains the next layer of capsules
    if route_mean:
        kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out, kernel_size[0]* kernel_size[1]* kernel_size[2], ch, mdim2 + 1])
        kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
        capsules = create_dense_caps((kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]), channels, name,
                                     route_min=route_min, ch_same_w=ch_same_w)
    else:
        kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out, kernel_size[0], kernel_size[1], kernel_size[2], ch, mdim2 + 1])
        capsules = create_dense_caps((kernels_reshaped[:, :, :, :, :, :-1], kernels_reshaped[:, :, :, :, :, -1:]),
                                     channels, name, subset_routing=subset_routing, route_min=route_min,
                                     coord_add=coord_add, rel_center=rel_center, ch_same_w=ch_same_w)

    # reshape capsules back into the 3d shape
    poses = tf.reshape(capsules[0][:, :, :mdim2], (batch_size, d_out, h_out, w_out, channels, mdim2))
    activations = tf.reshape(capsules[1], (batch_size, d_out, h_out, w_out, channels, 1))

    return poses, activations


def layer_shape(layer):
    """
    Returns a string with the shape of a capsule layer (pose matrices and activations)
    """
    return str(layer[0].get_shape()) + ' ' + str(layer[1].get_shape())