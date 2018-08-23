#Copy paste from https://github.com/zadaianchuk/HyperGradientDescent/blob/master/SGDN_HD_optimizer.py
#Hypergradient Descent Optimizer


from __future__ import division

import tensorflow as tf

class MomentumSGDHDOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, alpha_0, beta =10**(-7), name="HGD", mu=0.95, type_of_learning_rate ="global"):
        super(MomentumSGDHDOptimizer, self).__init__(beta, name=name)
        self._mu = mu
        self._alpha_0 = alpha_0
        self._beta = beta
        self._type = type_of_learning_rate


    def minimize(self, loss, global_step):

        # Algo params as constant tensors
        mu = tf.convert_to_tensor(self._mu, dtype=tf.float32)
        alpha_0=tf.convert_to_tensor(self._alpha_0, dtype=tf.float32)
        beta=tf.convert_to_tensor(self._beta, dtype=tf.float32)

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
        # moving average estimation
        ms = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "m", "m")
            for var in var_list]

        # update moving averages of the stochastic gradient:
        grads = tf.gradients(loss, var_list)
        m_updates = [m.assign(mu*m + (1.0-mu)*g) for m, g in zip(ms, grads)]

        #update of learning rate alpha, it is the main difference between SGD with Nesterov momentum
        #and its hypergradient version
        if self._type == "global":
            hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d,g in zip(ds, grads)])
            alphas_update = [alpha.assign(alpha-beta*hypergrad)]
        else:
            hypergrads = [tf.multiply(d,g) for d,g in zip(ds, grads)]
            alphas_update = [alpha.assign(alpha-beta*hypergrad) for alpha,hypergrad in zip(alphas,hypergrads)]

        # update step directions
        with tf.control_dependencies(m_updates+alphas_update):  #we want to be sure that alphas calculated using previous step directions
            ds_updates=[d.assign(-(mu*m + (1.0-mu)*g)) for (m,d,g) in zip(ms,ds,grads)]

        # update parameters of the model
        with tf.control_dependencies(ds_updates):
                if self._type == "global":
                    alpha_norm = alpha
                    variable_updates = [v.assign_add(alpha*d) for v, d in zip(var_list, ds)]
                else:
                    alpha_norm = sum([tf.reduce_mean(alpha**2) for alpha in alphas])
                    variable_updates = [v.assign_add(alpha*d) for v,d,alpha in zip(var_list, ds,alphas)]
                global_step.assign_add(1)

                #add summuries  (track alphas changes)
                with tf.name_scope("summaries"):
                    with tf.name_scope("per_iteration"):
                        alpha_sum=tf.summary.scalar("alpha", alpha_norm, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
        return tf.group(*variable_updates)