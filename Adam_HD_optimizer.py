#Copy paste from https://github.com/zadaianchuk/HyperGradientDescent/blob/master/Adam_HD_optimizer.py
#Hypergradient Descent Optimizer




from __future__ import division

import tensorflow as tf

class AdamHDOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, alpha_0, beta =10**(-7), name="HGD", mu=0.99, eps = 10**(-8),type_of_learning_rate ="global"):
        super(AdamHDOptimizer, self).__init__(beta, name=name)

        self._mu = mu
        self._alpha_0 = alpha_0
        self._beta = beta
        self._eps = eps
        self._type = type_of_learning_rate


    def minimize(self, loss, global_step):

        # Algo params as constant tensors
        mu = tf.convert_to_tensor(self._mu, dtype=tf.float32)
        alpha_0=tf.convert_to_tensor(self._alpha_0, dtype=tf.float32)
        beta=tf.convert_to_tensor(self._beta, dtype=tf.float32)
        eps = tf.convert_to_tensor(self._eps, dtype=tf.float32)

        var_list = tf.trainable_variables()

        # create and retrieve slot variables for:
        # direction of previous step
        ds = [self._get_or_make_slot(var,
                  tf.constant(0.0, tf.float32, var.get_shape()), "direction", "direction")
                  for var in var_list]
        # current learning_rate alpha
        if self._type == "global":
            alpha = self._get_or_make_slot(alpha_0, alpha_0, "learning_rate", "learning_rate")
        else:
            alphas = [self._get_or_make_slot(var,
                      tf.constant(self._alpha_0, tf.float32, var.get_shape()), "learning_rates", "learning_rates")
                      for var in var_list]
        #  moving average estimation
        ms = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "m", "m")
            for var in var_list]
        vs = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "v", "v")
            for var in var_list]
        # power of mu for bias-corrected first and second moment estimate
        mu_power = tf.get_variable("mu_power", shape=(), dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(1.0))

        # update moving averages of first and second moment:
        grads = tf.gradients(loss, var_list)
        grads_squared = [tf.square(g) for g in grads]
        m_updates = [m.assign(mu*m + (1.0-mu)*g) for m, g in zip(ms, grads)] #new means
        v_updates = [v.assign(mu*v + (1.0-mu)*g2) for v, g2 in zip(vs, grads_squared)]
        mu_power_update = [tf.assign(mu_power,tf.multiply(mu_power,mu))]
        # bais correction of the estimates
        with tf.control_dependencies(v_updates+m_updates+mu_power_update):
            ms_hat = [tf.divide(m,tf.constant(1.0) - mu_power) for m in ms]
            vs_hat = [tf.divide(v,tf.constant(1.0) - mu_power) for v in vs]

        #update of learning rate alpha, main difference between ADAM and ADAM-HD
        if self._type == "global":
            hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d,g in zip(ds, grads)])
            alphas_update = [alpha.assign(alpha-beta*hypergrad)]
        else:
            hypergrads = [tf.multiply(d,g) for d,g in zip(ds, grads)]
            alphas_update = [alpha.assign(alpha-beta*hypergrad) for alpha,hypergrad in zip(alphas,hypergrads)]

        # update step directions
        with tf.control_dependencies(alphas_update): #we want to be sure that alphas calculated using previous step directions
            ds_updates=[d.assign(-tf.divide(m, tf.sqrt(v) + self._eps)) for (m,v,d) in zip(ms_hat,vs_hat,ds)]

        # update parameters of the model
        with tf.control_dependencies(ds_updates):
                if self._type == "global":
                    dirs = [alpha*d for  d in ds]
                    alpha_norm = alpha
                else:
                    dirs = [alpha*d for  d, alpha in zip(ds,alphas)]
                    alpha_norm = sum([tf.reduce_mean(alpha**2) for alpha in alphas])
                variable_updates = [v.assign_add(d) for v, d in zip(var_list, dirs)]
                global_step.assign_add(1)
                # add summaries  (track alphas changes)
                with tf.name_scope("summaries"):
                    with tf.name_scope("per_iteration"):
                        alpha_norm_sum=tf.summary.scalar("alpha", alpha_norm, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
        return tf.group(*variable_updates)