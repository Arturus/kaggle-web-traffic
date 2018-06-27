import tensorflow as tf

from feeder import VarFeeder
from enum import Enum
from typing import List, Iterable
import numpy as np
import pandas as pd


class ModelMode(Enum):
    TRAIN = 0
    EVAL = 1,
    PREDICT = 2


class Split:
    def __init__(self, test_set: List[tf.Tensor], train_set: List[tf.Tensor], test_size: int, train_size: int):
        self.test_set = test_set
        self.train_set = train_set
        self.test_size = test_size
        self.train_size = train_size


class Splitter:
    def cluster_pages(self, cluster_idx: tf.Tensor):
        """
        Shuffles pages so all user_agents of each unique pages stays together in a shuffled list
        :param cluster_idx: Tensor[uniq_pages, n_agents], each value is index of pair (uniq_page, agent) in other page tensors
        :return: list of page indexes for use in a global page tensors
        """
        size = cluster_idx.shape[0].value
        random_idx = tf.random_shuffle(tf.range(0, size, dtype=tf.int32), self.seed)
        shuffled_pages = tf.gather(cluster_idx, random_idx)
        # Drop non-existent (uniq_page, agent) pairs. Non-existent pair has index value = -1
        mask = shuffled_pages >= 0
        page_idx = tf.boolean_mask(shuffled_pages, mask)
        return page_idx

    def __init__(self, tensors: List[tf.Tensor], cluster_indexes: tf.Tensor, n_splits, seed, train_sampling=1.0,
                 test_sampling=1.0):
        size = tensors[0].shape[0].value
        self.seed = seed
        clustered_index = self.cluster_pages(cluster_indexes)
        index_len = tf.shape(clustered_index)[0]
        assert_op = tf.assert_equal(index_len, size, message='n_pages is not equals to size of clustered index')
        with tf.control_dependencies([assert_op]):
            split_nitems = int(round(size / n_splits))
            split_size = [split_nitems] * n_splits
            split_size[-1] = size - (n_splits - 1) * split_nitems
            splits = tf.split(clustered_index, split_size)
            complements = [tf.random_shuffle(tf.concat(splits[:i] + splits[i + 1:], axis=0), seed) for i in
                           range(n_splits)]
            splits = [tf.random_shuffle(split, seed) for split in splits]

        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            test_size = split_size[i]
            train_size = size - test_size
            test_sampled_size = int(round(test_size * test_sampling))
            train_sampled_size = int(round(train_size * train_sampling))
            test_idx = splits[i][:test_sampled_size]
            train_idx = complements[i][:train_sampled_size]
            test_set = [tf.gather(tensor, test_idx, name=mk_name('test', tensor)) for tensor in tensors]
            tran_set = [tf.gather(tensor, train_idx, name=mk_name('train', tensor)) for tensor in tensors]
            return Split(test_set, tran_set, test_sampled_size, train_sampled_size)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class FakeSplitter:
    def __init__(self, tensors: List[tf.Tensor], n_splits, seed, test_sampling=1.0):
        total_pages = tensors[0].shape[0].value
        n_pages = int(round(total_pages * test_sampling))

        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            idx = tf.random_shuffle(tf.range(0, n_pages, dtype=tf.int32), seed + i)
            train_tensors = [tf.gather(tensor, idx, name=mk_name('shfl', tensor)) for tensor in tensors]
            if test_sampling < 1.0:
                sampled_idx = idx[:n_pages]
                test_tensors = [tf.gather(tensor, sampled_idx, name=mk_name('shfl_test', tensor)) for tensor in tensors]
            else:
                test_tensors = train_tensors
            return Split(test_tensors, train_tensors, n_pages, total_pages)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class InputPipe:
    def cut(self, hits, start, end):
        """
        Cuts [start:end] diapason from input data
        :param hits: hits timeseries
        :param start: start index
        :param end: end index
        :return: tuple (train_hits, test_hits, dow, lagged_hits)
        """
        # Pad hits to ensure we have enough array length for prediction
        hits = tf.concat([hits, tf.fill([self.predict_window], np.NaN)], axis=0)
        cropped_hit = hits[start:end]

        # cut day of week
        cropped_dow = self.inp.dow[start:end]

        # Cut lagged hits
        # gather() accepts only int32 indexes
        cropped_lags = tf.cast(self.inp.lagged_ix[start:end], tf.int32)
        # Mask for -1 (no data) lag indexes
        lag_mask = cropped_lags < 0
        # Convert -1 to 0 for gather(), it don't accept anything exotic
        cropped_lags = tf.maximum(cropped_lags, 0)
        # Translate lag indexes to hit values
        lagged_hit = tf.gather(hits, cropped_lags)
        # Convert masked (see above) or NaN lagged hits to zeros
        lag_zeros = tf.zeros_like(lagged_hit)
        lagged_hit = tf.where(lag_mask | tf.is_nan(lagged_hit), lag_zeros, lagged_hit)

        # Split for train and test
        x_hits, y_hits = tf.split(cropped_hit, [self.train_window, self.predict_window], axis=0)

        # Convert NaN to zero in for train data
        x_hits = tf.where(tf.is_nan(x_hits), tf.zeros_like(x_hits), x_hits)
        return x_hits, y_hits, cropped_dow, lagged_hit

    def cut_train(self, hits, *args):
        """
        Cuts a segment of time series for training. Randomly chooses starting point.
        :param hits: hits timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        n_days = self.predict_window + self.train_window
        # How much free space we have to choose starting day
        free_space = self.inp.data_days - n_days - self.back_offset - self.start_offset
        if self.verbose:
            lower_train_start = self.inp.data_start + pd.Timedelta(self.start_offset, 'D')
            lower_test_end = lower_train_start + pd.Timedelta(n_days, 'D')
            lower_test_start = lower_test_end - pd.Timedelta(self.predict_window, 'D')
            upper_train_start = self.inp.data_start + pd.Timedelta(free_space - 1, 'D')
            upper_test_end = upper_train_start + pd.Timedelta(n_days, 'D')
            upper_test_start = upper_test_end - pd.Timedelta(self.predict_window, 'D')
            print(f"Free space for training: {free_space} days.")
            print(f" Lower train {lower_train_start}, prediction {lower_test_start}..{lower_test_end}")
            print(f" Upper train {upper_train_start}, prediction {upper_test_start}..{upper_test_end}")
        # Random starting point
        offset = tf.random_uniform((), self.start_offset, free_space, dtype=tf.int32, seed=self.rand_seed)
        end = offset + n_days
        # Cut all the things
        return self.cut(hits, offset, end) + args

    def cut_eval(self, hits, *args):
        """
        Cuts segment of time series for evaluation.
        Always cuts train_window + predict_window length segment beginning at start_offset point
        :param hits: hits timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        end = self.start_offset + self.train_window + self.predict_window
        return self.cut(hits, self.start_offset, end) + args

    def reject_filter(self, x_hits, y_hits, *args):
        """
        Rejects timeseries having too many zero datapoints (more than self.max_train_empty)
        """
        if self.verbose:
            print("max empty %d train %d predict" % (self.max_train_empty, self.max_predict_empty))
        zeros_x = tf.reduce_sum(tf.to_int32(tf.equal(x_hits, 0.0)))
        keep = zeros_x <= self.max_train_empty
        return keep

    def make_features(self, x_hits, y_hits, dow, lagged_hits, pf_agent, pf_country, pf_site, page_ix,
                      page_popularity, year_autocorr, quarter_autocorr):
        """
        Main method. Assembles input data into final tensors
        """
        # Split day of week to train and test
        x_dow, y_dow = tf.split(dow, [self.train_window, self.predict_window], axis=0)

        # Normalize hits
        mean = tf.reduce_mean(x_hits)
        std = tf.sqrt(tf.reduce_mean(tf.squared_difference(x_hits, mean)))
        norm_x_hits = (x_hits - mean) / std
        norm_y_hits = (y_hits - mean) / std
        norm_lagged_hits = (lagged_hits - mean) / std

        # Split lagged hits to train and test
        x_lagged, y_lagged = tf.split(norm_lagged_hits, [self.train_window, self.predict_window], axis=0)

        # Combine all page features into single tensor
        stacked_features = tf.stack([page_popularity, quarter_autocorr, year_autocorr])
        flat_page_features = tf.concat([pf_agent, pf_country, pf_site, stacked_features], axis=0)
        page_features = tf.expand_dims(flat_page_features, 0)

        # Train features
        x_features = tf.concat([
            # [n_days] -> [n_days, 1]
            tf.expand_dims(norm_x_hits, -1),
            x_dow,
            x_lagged,
            # Stretch page_features to all training days
            # [1, features] -> [n_days, features]
            tf.tile(page_features, [self.train_window, 1])
        ], axis=1)

        # Test features
        y_features = tf.concat([
            # [n_days] -> [n_days, 1]
            y_dow,
            y_lagged,
            # Stretch page_features to all testing days
            # [1, features] -> [n_days, features]
            tf.tile(page_features, [self.predict_window, 1])
        ], axis=1)

        return x_hits, x_features, norm_x_hits, x_lagged, y_hits, y_features, norm_y_hits, mean, std, flat_page_features, page_ix

    def __init__(self, inp: VarFeeder, features: Iterable[tf.Tensor], n_pages: int, mode: ModelMode, n_epoch=None,
                 batch_size=127, runs_in_burst=1, verbose=True, predict_window=60, train_window=500,
                 train_completeness_threshold=1, predict_completeness_threshold=1, back_offset=0,
                 train_skip_first=0, rand_seed=None):
        """
        Create data preprocessing pipeline
        :param inp: Raw input data
        :param features: Features tensors (subset of data in inp)
        :param n_pages: Total number of pages
        :param mode: Train/Predict/Eval mode selector
        :param n_epoch: Number of epochs. Generates endless data stream if None
        :param batch_size:
        :param runs_in_burst: How many batches can be consumed at short time interval (burst). Multiplicator for prefetch()
        :param verbose: Print additional information during graph construction
        :param predict_window: Number of days to predict
        :param train_window: Use train_window days for traning
        :param train_completeness_threshold: Percent of zero datapoints allowed in train timeseries.
        :param predict_completeness_threshold: Percent of zero datapoints allowed in test/predict timeseries.
        :param back_offset: Don't use back_offset days at the end of timeseries
        :param train_skip_first: Don't use train_skip_first days at the beginning of timeseries
        :param rand_seed:

        """
        self.n_pages = n_pages
        self.inp = inp
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.back_offset = back_offset
        if verbose:
            print("Mode:%s, data days:%d, Data start:%s, data end:%s, features end:%s " % (
            mode, inp.data_days, inp.data_start, inp.data_end, inp.features_end))

        if mode == ModelMode.TRAIN:
            # reserve predict_window at the end for validation
            assert inp.data_days - predict_window > predict_window + train_window, \
                "Predict+train window length (+predict window for validation) is larger than total number of days in dataset"
            self.start_offset = train_skip_first
        elif mode == ModelMode.EVAL or mode == ModelMode.PREDICT:
            self.start_offset = inp.data_days - train_window - back_offset
            if verbose:
                train_start = inp.data_start + pd.Timedelta(self.start_offset, 'D')
                eval_start = train_start + pd.Timedelta(train_window, 'D')
                end = eval_start + pd.Timedelta(predict_window - 1, 'D')
                print("Train start %s, predict start %s, end %s" % (train_start, eval_start, end))
            assert self.start_offset >= 0

        self.train_window = train_window
        self.predict_window = predict_window
        self.attn_window = train_window - predict_window + 1
        self.max_train_empty = int(round(train_window * (1 - train_completeness_threshold)))
        self.max_predict_empty = int(round(predict_window * (1 - predict_completeness_threshold)))
        self.mode = mode
        self.verbose = verbose

        # Reserve more processing threads for eval/predict because of larger batches
        num_threads = 3 if mode == ModelMode.TRAIN else 6

        # Choose right cutter function for current ModelMode
        cutter = {ModelMode.TRAIN: self.cut_train, ModelMode.EVAL: self.cut_eval, ModelMode.PREDICT: self.cut_eval}
        # Create dataset, transform features and assemble batches
        root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
        batch = (root_ds
                 .map(cutter[mode])
                 .filter(self.reject_filter)
                 .map(self.make_features, num_parallel_calls=num_threads)
                 .batch(batch_size)
                 .prefetch(runs_in_burst * 2)
                 )

        self.iterator = batch.make_initializable_iterator()
        it_tensors = self.iterator.get_next()

        # Assign all tensors to class variables
        self.true_x, self.time_x, self.norm_x, self.lagged_x, self.true_y, self.time_y, self.norm_y, self.norm_mean, \
        self.norm_std, self.page_features, self.page_ix = it_tensors

        self.encoder_features_depth = self.time_x.shape[2].value

    def load_vars(self, session):
        self.inp.restore(session)

    def init_iterator(self, session):
        session.run(self.iterator.initializer)


def page_features(inp: VarFeeder):
    return (inp.hits, inp.pf_agent, inp.pf_country, inp.pf_site,
            inp.page_ix, inp.page_popularity, inp.year_autocorr, inp.quarter_autocorr)
