from collections import UserList, UserDict
from typing import Union, Iterable, Tuple, Dict, Any

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os.path


def _meta_file(path):
    return os.path.join(path, 'feeder_meta.pkl')


class VarFeeder:
    """
    Helper to avoid feed_dict and manual batching. Maybe I had to use TFRecords instead.
    Builds temporary TF graph, injects variables into, and saves variables to TF checkpoint.
    In a train time, variables can be built by build_vars() and content restored by FeederVars.restore()
    """
    def __init__(self, path: str,
                 tensor_vars: Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                 plain_vars: Dict[str, Any] = None):
        """
        :param path: dir to store data
        :param tensor_vars: Variables to save as Tensors (pandas DataFrames/Series or numpy arrays)
        :param plain_vars: Variables to save as Python objects
        """
        tensor_vars = tensor_vars or dict()

        def get_values(v):
            v = v.values if hasattr(v, 'values') else v
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            return v

        values = [get_values(var) for var in tensor_vars.values()]

        self.shapes = [var.shape for var in values]
        self.dtypes = [v.dtype for v in values]
        self.names = list(tensor_vars.keys())
        self.path = path
        self.plain_vars = plain_vars

        if not os.path.exists(path):
            os.mkdir(path)

        with open(_meta_file(path), mode='wb') as file:
            pickle.dump(self, file)

        with tf.Graph().as_default():
            tensor_vars = self._build_vars()
            placeholders = [tf.placeholder(tf.as_dtype(dtype), shape=shape) for dtype, shape in
                            zip(self.dtypes, self.shapes)]
            assigners = [tensor_var.assign(placeholder) for tensor_var, placeholder in
                         zip(tensor_vars, placeholders)]
            feed = {ph: v for ph, v in zip(placeholders, values)}
            saver = tf.train.Saver(self._var_dict(tensor_vars), max_to_keep=1)
            init = tf.global_variables_initializer()

            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                sess.run(init)
                sess.run(assigners, feed_dict=feed)
                save_path = os.path.join(path, 'feeder.cpt')
                saver.save(sess, save_path, write_meta_graph=False, write_state=False)

    def _var_dict(self, variables):
        return {name: var for name, var in zip(self.names, variables)}

    def _build_vars(self):
        def make_tensor(shape, dtype, name):
            tf_type = tf.as_dtype(dtype)
            if tf_type == tf.string:
                empty = ''
            elif tf_type == tf.bool:
                empty = False
            else:
                empty = 0
            init = tf.constant(empty, shape=shape, dtype=tf_type)
            return tf.get_local_variable(name=name, initializer=init, dtype=tf_type)

        with tf.device("/cpu:0"):
            with tf.name_scope('feeder_vars'):
                return [make_tensor(shape, dtype, name) for shape, dtype, name in
                        zip(self.shapes, self.dtypes, self.names)]

    def create_vars(self):
        """
        Builds variable list to use in current graph. Should be called during graph building stage
        :return: variable list with additional restore and create_saver methods
        """
        return FeederVars(self._var_dict(self._build_vars()), self.plain_vars, self.path)

    @staticmethod
    def read_vars(path):
        with open(_meta_file(path), mode='rb') as file:
            feeder = pickle.load(file)
        assert feeder.path == path
        return feeder.create_vars()


class FeederVars(UserDict):
    def __init__(self, tensors: dict, plain_vars: dict, path):
        variables = dict(tensors)
        if plain_vars:
            variables.update(plain_vars)
        super().__init__(variables)
        self.path = path
        self.saver = tf.train.Saver(tensors, name='varfeeder_saver')
        for var in variables:
            if var not in self.__dict__:
                self.__dict__[var] = variables[var]

    def restore(self, session):
        """
        Restores variable content
        :param session: current session
        :return: variable list
        """
        self.saver.restore(session, os.path.join(self.path, 'feeder.cpt'))
        return self
