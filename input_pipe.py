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
    """
    This is the splitter used when side_split
    (vs. FakeSplitter when not side_split [when forward_split])
    
    Is typical train-test split
    """
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
        assert_op = tf.assert_equal(index_len, size, message='N_time_series is not equals to size of clustered index')
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
            
#            print(test_size)
#            print(train_size)
#            print(test_sampled_size)
#            print(train_sampled_size)   
#            print(test_idx)
#            print(train_idx)
            #When doing --side_split validation option, was getting a type error
            #when creating test_set, tran_set list comprehensions: change dtype here for idx
            test_idx = tf.cast(test_idx, tf.int32)
            train_idx = tf.cast(train_idx, tf.int32)
            
            test_idx = tf.Print(test_idx, ['test_idx',tf.shape(test_idx),test_idx])
            train_idx = tf.Print(train_idx, ['train_idx',tf.shape(train_idx),train_idx])
            """48354
            96709
            48354
            96709
            Tensor("strided_slice_1:0", shape=(48354,), dtype=float32, device=/device:CPU:0)
            Tensor("strided_slice_2:0", shape=(96709,), dtype=float32, device=/device:CPU:0)"""     
            test_set = [tf.gather(tensor, test_idx, name=mk_name('test', tensor)) for tensor in tensors]
            tran_set = [tf.gather(tensor, train_idx, name=mk_name('train', tensor)) for tensor in tensors]
#            print(test_set)
#            print(tran_set)
            return Split(test_set, tran_set, test_sampled_size, train_sampled_size)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class FakeSplitter:
    """
    This is the splitter used when forward_split
    (vs. Splitter when not forward_split [when side_split])
    
    Is typical train-test split
    """    
    def __init__(self, tensors: List[tf.Tensor], n_splits, seed, test_sampling=1.0):
        total_series = tensors[0].shape[0].value
        N_time_series = int(round(total_series * test_sampling))

        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            idx = tf.random_shuffle(tf.range(0, N_time_series, dtype=tf.int32), seed + i)
            train_tensors = [tf.gather(tensor, idx, name=mk_name('shfl', tensor)) for tensor in tensors]
            if test_sampling < 1.0: #Only use subset of time series = test_sampling
                sampled_idx = idx[:N_time_series]
                test_tensors = [tf.gather(tensor, sampled_idx, name=mk_name('shfl_test', tensor)) for tensor in tensors]
            else:
                test_tensors = train_tensors
            return Split(test_tensors, train_tensors, N_time_series, total_series)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class InputPipe:
#    def randomize_window_sizes(self, *args):
#        self.horizon_window_size = tf.random_uniform((), 7, 60, dtype=tf.int32)
#        self.history_window_size = tf.random_uniform((), 7, 366, dtype=tf.int32)
#        self.attn_window = self.history_window_size - self.horizon_window_size + 1
#        self.max_train_empty = tf.cast(tf.round(tf.multiply(tf.cast(self.history_window_size,tf.float32),(1 - self.train_completeness_threshold))),tf.int32)
#        self.max_predict_empty = tf.cast(tf.round(tf.multiply(tf.cast(self.horizon_window_size,tf.float32),(1 - self.predict_completeness_threshold))),tf.int32)
#        return args
        
    def cut(self, counts, start, end):
        """
        Cuts [start:end] diapason from input data
        :param counts: counts timeseries
        :param start: start index
        :param end: end index
        :return: tuple (train_counts, test_counts, lagged_counts, [subset of: dow,woy,moy,doy,year,holidays])
        """
        
        # Pad counts to ensure we have enough array length for prediction
        counts = tf.concat([counts, tf.fill([self.horizon_window_size], np.NaN)], axis=0)
        cropped_count = counts[start:end]
#        cropped_count = tf.Print(cropped_count,['INPUT PIPE > CUT > cropped_count',tf.shape(cropped_count), 'start', start, 'end', end])
#        cropped_count = tf.Print(cropped_count,['self.history_window_size', self.history_window_size, 'self.horizon_window_size', self.horizon_window_size])
        
        # =============================================================================
        # Ordinal periodic variables
        # which features are here depends on what the sampling period is for the data
        # =============================================================================
        if self.sampling_period=='daily':
            cropped_dow = self.inp.dow[start:end]
            cropped_woy = self.inp.woy[start:end]
            cropped_doy = self.inp.doy[start:end]
            cropped_holidays = self.inp.holidays[start:end]
#            cropped_moy = 0*cropped_dow #Month information is alreayd contained in week information. COuld incude anyway to be explicit, but for now do not use as a feature
        elif self.sampling_period=='weekly':
            cropped_woy = self.inp.woy[start:end]
#            cropped_dow = 0*cropped_woy
#            cropped_moy = 0*cropped_woy
        elif self.sampling_period=='monthly':
            cropped_moy = self.inp.moy[start:end]
#            cropped_dow = 0*cropped_moy
#            cropped_woy = 0*cropped_moy            
            
        #ANd use year as a feature to get long term trend
        cropped_year = self.inp.year[start:end]

        
        # =============================================================================
        # Other features that are also time-varying
        # that can be used, which depend on the choice of feature_set
        # self.features_set = features_set
        # =============================================================================        
        
        #If used Arturius' original feature set then will include the lagged data:
#        if self.features_set == 'arturius':
        if self.features_set=='arturius':
            # Cut lagged counts
            # gather() accepts only int32 indexes
            cropped_lags = tf.cast(self.inp.lagged_ix[start:end], tf.int32)
            # Mask for -1 (no data) lag indexes
            lag_mask = cropped_lags < 0
            # Convert -1 to 0 for gather(), it don't accept anything exotic
            cropped_lags = tf.maximum(cropped_lags, 0)
            # Translate lag indexes to count values
            lagged_count = tf.gather(counts, cropped_lags)
            # Convert masked (see above) or NaN lagged counts to zeros
            lag_zeros = tf.zeros_like(lagged_count)
            lagged_count = tf.where(lag_mask | tf.is_nan(lagged_count), lag_zeros, lagged_count)



        #Will always have the count series (the series we predict on):
        # Split for train and test
        x_counts, y_counts = tf.split(cropped_count, [self.history_window_size, self.horizon_window_size], axis=0)

        # Convert NaN to zero in for train data
        x_counts = tf.where(tf.is_nan(x_counts), tf.zeros_like(x_counts), x_counts)



        if self.features_set=='arturius':
            if self.sampling_period=='daily':
                return x_counts, y_counts, lagged_count, cropped_dow, cropped_woy, cropped_doy, cropped_year, cropped_holidays
            if self.sampling_period=='weekly':
                return x_counts, y_counts, lagged_count, cropped_woy, cropped_year
            if self.sampling_period=='monthly':
                return x_counts, y_counts, lagged_count, cropped_moy, cropped_year
            
            
            
        elif self.features_set=='full':
            if self.sampling_period=='daily':
                return x_counts, y_counts, cropped_dow, cropped_woy, cropped_doy, cropped_year, cropped_holidays
            if self.sampling_period=='weekly':
                return x_counts, y_counts, cropped_woy, cropped_year, cropped_holidays
            if self.sampling_period=='monthly':
                return x_counts, y_counts, cropped_moy, cropped_year, cropped_holidays
        else:
            print(self.features_set)
            raise Exception('problem with features_set')


    def cut_train(self, counts, *args):
        """
        Cuts a segment of time series for training. Randomly chooses starting point.
        :param counts: counts timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
#        def randomize_window_sizes():#, *args):
#            self.horizon_window_size = tf.random_uniform((), 7, 60, dtype=tf.int32)
#            self.history_window_size = tf.random_uniform((), 7, 366, dtype=tf.int32)
#            self.attn_window = self.history_window_size - self.horizon_window_size + 1
#            self.max_train_empty = tf.cast(tf.round(tf.multiply(tf.cast(self.history_window_size,tf.float32),(1 - self.train_completeness_threshold))),tf.int32)
#            self.max_predict_empty = tf.cast(tf.round(tf.multiply(tf.cast(self.horizon_window_size,tf.float32),(1 - self.predict_completeness_threshold))),tf.int32)
#            #return args        
#        randomize_window_sizes()        
        
        n_timesteps = self.horizon_window_size + self.history_window_size
        # How much free space we have to choose starting day
        free_space = self.inp.data_timesteps - n_timesteps - self.back_offset - self.start_offset
        if self.verbose:
            #!!!!!! doesn't really matter since this is just printout, but would need to change for WEEKLY / MONTHLY
            lower_train_start = self.inp.data_start + pd.Timedelta(self.start_offset, 'D')
            lower_test_end = lower_train_start + pd.Timedelta(n_timesteps, 'D')
            lower_test_start = lower_test_end - pd.Timedelta(self.horizon_window_size, 'D')
            upper_train_start = self.inp.data_start + pd.Timedelta(free_space - 1, 'D')
            upper_test_end = upper_train_start + pd.Timedelta(n_timesteps, 'D')
            upper_test_start = upper_test_end - pd.Timedelta(self.horizon_window_size, 'D')
            print(f"Free space for training: {free_space} days.")
            print(f" Lower train {lower_train_start}, prediction {lower_test_start}..{lower_test_end}")
            print(f" Upper train {upper_train_start}, prediction {upper_test_start}..{upper_test_end}")
        # Random starting point
        offset = tf.random_uniform((), self.start_offset, free_space, dtype=tf.int32, seed=self.rand_seed)
#        offset = tf.Print(offset,['offset',tf.shape(offset),offset])
        end = offset + n_timesteps
        # Cut all the things
        return self.cut(counts, offset, end) + args

    def cut_eval(self, counts, *args):
        """
        Cuts segment of time series for evaluation.
        Always cuts history_window_size + horizon_window_size length segment beginning at start_offset point
        :param counts: counts timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        end = self.start_offset + self.history_window_size + self.horizon_window_size
        return self.cut(counts, self.start_offset, end) + args

    def reject_filter(self, x_counts, y_counts, *args):
        """
        Rejects timeseries having too many zero datapoints (more than self.max_train_empty)
        [by this point, NANs would have already been converted to 0's, this is is NAN's U 0's]
        """
        if self.verbose:
            print("max empty %d train %d predict" % (self.max_train_empty, self.max_predict_empty))
        zeros_x = tf.reduce_sum(tf.to_int32(tf.equal(x_counts, 0.0)))
        keep = zeros_x <= self.max_train_empty
        return keep





    def make_features(self, *args):        
        """
        Main method. Assembles input data into final tensors
        """
        # =============================================================================
        # Unpack the vars depending on which features_set - sampling_period
        # The order needs to match the output of the cut method.
        # cut_train and cut_eval return args + cut_output
        # the args are things like pf_agent, p
        # the cut_output is the same order as the return of the cut method.
        # =============================================================================
        print(args)
        if self.features_set == 'arturius':
            if self.sampling_period == 'daily':
                x_counts, y_counts, lagged_counts, dow, woy, doy, year, holidays, pf_agent, pf_country, pf_site, page_ix, count_median, year_autocorr, quarter_autocorr, count_pctl_100 = args
            elif self.sampling_period == 'weekly':
                x_counts, y_counts, lagged_counts, woy, year, holidays, pf_agent, pf_country, pf_site, page_ix, count_median, year_autocorr, quarter_autocorr, count_pctl_100 = args        
            elif self.sampling_period == 'monthly':
                x_counts, y_counts, lagged_counts, moy, year, holidays, pf_agent, pf_country, pf_site, page_ix, count_median, year_autocorr, quarter_autocorr, count_pctl_100 = args          
        #For now just use the same ...
#        count_pctl_0, count_pctl_5, count_pctl_25, count_pctl_75, count_pctl_95, count_pctl_100, count_variance)
        elif self.features_set == 'full':
            if self.sampling_period == 'daily': 
                x_counts, y_counts, dow, woy, doy, year, holidays, page_ix, count_median,\
                count_pctl_0, count_pctl_5, count_pctl_25, count_pctl_75, count_pctl_95, count_pctl_100, count_variance = args                
            elif self.sampling_period == 'weekly':
                x_counts, y_counts, woy, year, holidays, page_ix, count_median,\
                count_pctl_0, count_pctl_5, count_pctl_25, count_pctl_75, count_pctl_95, count_pctl_100, count_variance = args
            elif self.sampling_period == 'monthly':
                x_counts, y_counts, moy, year, holidays, page_ix, count_median,\
                count_pctl_0, count_pctl_5, count_pctl_25, count_pctl_75, count_pctl_95, count_pctl_100, count_variance = args
        
        # =============================================================================
        # Do train - predict splits
        # =============================================================================
        if self.sampling_period == 'daily':
            x_dow, y_dow = tf.split(dow, [self.history_window_size, self.horizon_window_size], axis=0)
            x_woy, y_woy = tf.split(woy, [self.history_window_size, self.horizon_window_size], axis=0)
            x_doy, y_doy = tf.split(doy, [self.history_window_size, self.horizon_window_size], axis=0)
            x_holidays, y_holidays = tf.split(holidays, [self.history_window_size, self.horizon_window_size], axis=0)
        elif self.sampling_period == 'weekly':
            x_woy, y_woy = tf.split(woy, [self.history_window_size, self.horizon_window_size], axis=0)
        elif self.sampling_period == 'monthly':
            x_moy, y_moy = tf.split(moy, [self.history_window_size, self.horizon_window_size], axis=0)

        #Already did a manual kind of scaling for year in make_features.py so don't need to normalize here...
        x_year, y_year = tf.split(year, [self.history_window_size, self.horizon_window_size], axis=0)
        x_year = tf.expand_dims(x_year,axis=1)
        y_year = tf.expand_dims(y_year,axis=1)

        # Normalize counts
        mean = tf.reduce_mean(x_counts)
        std = tf.sqrt(tf.reduce_mean(tf.squared_difference(x_counts, mean)))
        norm_x_counts = (x_counts - mean) / std
        norm_y_counts = (y_counts - mean) / std
        
        if self.features_set == 'arturius':
            norm_lagged_counts = (lagged_counts - mean) / std
            # Split lagged counts to train and test
            x_lagged, y_lagged = tf.split(norm_lagged_counts, [self.history_window_size, self.horizon_window_size], axis=0)            
            scalar_features = tf.stack([count_median, quarter_autocorr, year_autocorr, count_pctl_100])
            flat_features = tf.concat([pf_agent, pf_country, pf_site, scalar_features], axis=0) 
            series_features = tf.expand_dims(flat_features, 0)
        elif self.features_set == 'full':
            scalar_features = tf.stack([count_median, count_pctl_0, count_pctl_5, count_pctl_25, count_pctl_75, count_pctl_95, count_pctl_100, count_variance])
            #flat_features = tf.concat([pf_agent, pf_country, pf_site, scalar_features], axis=0) 
            flat_features = tf.concat([scalar_features], axis=0) 
            series_features = tf.expand_dims(flat_features, 0)
        
        
        
        
#            print(scalar_features) #4
#            print(flat_features) #18
#            print(series_features)
#            print([pf_agent, pf_country, pf_site]) #4, 7, 3   #the one hot encoded features
        

        #Any time dependent feature need to be split into x [train] and y [test]
        #the time INdependent features [constant per fixture] will just be tiled same way either way except diff lengths

        # Train features, depending on measurement frequency
        x_features = tf.expand_dims(norm_x_counts, -1) # [n_timesteps] -> [n_timesteps, 1]
        if self.sampling_period == 'daily':
            x_features = tf.concat([x_features, x_dow, x_woy, tf.expand_dims(x_doy,-1), x_year, x_holidays], axis=1)
        elif self.sampling_period == 'weekly':
            x_features = tf.concat([x_features, x_woy, x_year, x_holidays], axis=1)            
        elif self.sampling_period == 'monthly':
            x_features = tf.concat([x_features, x_moy, x_year, x_holidays], axis=1)             
        
        #Regardess of period/frequency will have below features:
        if self.features_set == 'arturius':
            x_features = tf.concat([x_features, x_lagged,
                                    # Stretch series_features to all training days
                                    # [1, features] -> [n_timesteps, features]
                                    tf.tile(series_features, [self.history_window_size, 1])], axis=1)
        elif self.features_set == 'full':
            x_features = tf.concat([x_features, 
                                    # Stretch series_features to all training days
                                    # [1, features] -> [n_timesteps, features]
                                    tf.tile(series_features, [self.history_window_size, 1])], axis=1)
                
                
        # Test features
        if self.sampling_period == 'daily':
            y_features = tf.concat([y_dow, y_woy, tf.expand_dims(y_doy,-1), y_year, y_holidays], axis=1)
        elif self.sampling_period == 'weekly':
            y_features = tf.concat([y_woy, y_year, y_holidays], axis=1)
        elif self.sampling_period == 'monthly':
            y_features = tf.concat([y_moy, y_year, y_holidays], axis=1)

        if self.features_set == 'arturius':
            y_features = tf.concat([y_features, y_lagged,
                                    # Stretch series_features to all testing days
                                    # [1, features] -> [n_timesteps, features]
                                    tf.tile(series_features, [self.horizon_window_size, 1])
                                    ], axis=1)
        elif self.features_set == 'full':
            y_features = tf.concat([y_features,
                                    # Stretch series_features to all testing days
                                    # [1, features] -> [n_timesteps, features]
                                    tf.tile(series_features, [self.horizon_window_size, 1])
                                    ], axis=1)                

#        print(x_features)
        
        #!!!!! why no lagged_y alnoe, only in y_features??? 
        #!!!! why no norm_y_counts ?????
        
        print('x_features')
        print(x_features)
        print(x_features.shape)
        if self.features_set == 'arturius':
            return x_counts, x_features, norm_x_counts, x_lagged, y_counts, y_features, norm_y_counts, mean, std, flat_features, page_ix
        if self.features_set == 'full':
            return x_counts, x_features, norm_x_counts, y_counts, y_features, norm_y_counts, mean, std, flat_features, page_ix
        #Must match up with setting self.XYZ = it_tensors below in __init__. 


    def __init__(self, features_set, sampling_period, inp: VarFeeder, features: Iterable[tf.Tensor], N_time_series: int, mode: ModelMode, n_epoch=None,
                 batch_size=127, runs_in_burst=1, verbose=True, horizon_window_size=60, history_window_size=500,
                 train_completeness_threshold=1, predict_completeness_threshold=1, back_offset=0,
                 train_skip_first=0, rand_seed=None):
        """
        Create data preprocessing pipeline
        features_set  -  arturius, simple, full, full_w_context
        sampling_period  -  daily, weekly, monthly
        :param inp: Raw input data
        :param features: Features tensors (subset of data in inp)
        :param N_time_series: Total number of pages
        :param mode: Train/Predict/Eval mode selector
        :param n_epoch: Number of epochs. Generates endless data stream if None
        :param batch_size:
        :param runs_in_burst: How many batches can be consumed at short time interval (burst). Multiplicator for prefetch()
        :param verbose: Print additional information during graph construction
        :param horizon_window_size: Number of days to predict
        :param history_window_size: Use history_window_size days for traning
        :param train_completeness_threshold: Percent of zero datapoints allowed in train timeseries.
        :param predict_completeness_threshold: Percent of zero datapoints allowed in test/predict timeseries.
        :param back_offset: Don't use back_offset days at the end of timeseries
        :param train_skip_first: Don't use train_skip_first days at the beginning of timeseries
        :param rand_seed:

        """
        
        self.features_set = features_set
        self.sampling_period = sampling_period
        
        self.N_time_series = N_time_series
        self.inp = inp
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.back_offset = back_offset
        if verbose:
            print("Mode:%s, data days:%d, Data start:%s, data end:%s, features end:%s " % (
            mode, inp.data_timesteps, inp.data_start, inp.data_end, inp.features_end))

        if mode == ModelMode.TRAIN:
            # reserve horizon_window_size at the end for validation
            assert inp.data_timesteps - horizon_window_size > horizon_window_size + history_window_size, \
                "Predict+train window length (+predict window for validation) is larger than total number of days in dataset"
            self.start_offset = train_skip_first
        elif mode == ModelMode.EVAL or mode == ModelMode.PREDICT:
            print('inp.data_timesteps',inp.data_timesteps)
            print('history_window_size',history_window_size)
            print('back_offset',back_offset)
            self.start_offset = inp.data_timesteps - history_window_size - back_offset #!!!!!
            if verbose:
                train_start = inp.data_start + pd.Timedelta(self.start_offset, 'D')
                eval_start = train_start + pd.Timedelta(history_window_size, 'D')
                end = eval_start + pd.Timedelta(horizon_window_size - 1, 'D')
                print("Train start %s, predict start %s, end %s" % (train_start, eval_start, end))
            assert self.start_offset >= 0

        self.history_window_size = history_window_size #!!!!!!!!!!!random resize
        self.horizon_window_size = horizon_window_size#!!!!!!!!!!!random resize
        self.attn_window = history_window_size - horizon_window_size + 1#!!!!!!!!!!!random resize
        #For train empty, if max_train_empty=history, then on data with many missing values, 
        #you can get all NAN series, which then causes NANs in inp.time_x and 
        #destroys the encoded_state and kills everything. You need to have at 
        #least 1 valid value in the history window, so do min(history-1, xxxxx)
        self.max_train_empty = min(history_window_size-1, int(np.floor(history_window_size * (1 - train_completeness_threshold))))
        self.max_predict_empty = int(np.floor(horizon_window_size * (1 - predict_completeness_threshold)))
        self.mode = mode
        self.verbose = verbose
        
        self.train_completeness_threshold = train_completeness_threshold
        self.predict_completeness_threshold = predict_completeness_threshold


        print('max_train_empty',self.max_train_empty)
        print('max_predict_empty',self.max_predict_empty)
        print('history_window_size',self.history_window_size)
        print('horizon_window_size',self.horizon_window_size)
        print('attn_window',self.attn_window)

        
        def random_draw_new_window_sizes():
            history = np.random.randint(low=7,high=120+1)
            horizon = np.random.randint(low=7,high=60+1)        
            self.history_window_size = history
            self.horizon_window_size = horizon
            self.attn_window = history - horizon + 1
            self.max_train_empty = min(history_window_size-1, int(np.floor(history_window_size * (1 - train_completeness_threshold))))
            self.max_predict_empty = int(np.floor(horizon * (1 - self.predict_completeness_threshold)))
    
    
        
        # Reserve more processing threads for eval/predict because of larger batches
        num_threads = 3 if mode == ModelMode.TRAIN else 6

        # Choose right cutter function for current ModelMode
        cutter = {ModelMode.TRAIN: self.cut_train, ModelMode.EVAL: self.cut_eval, ModelMode.PREDICT: self.cut_eval}
        # Create dataset, transform features and assemble batches
        #features is a list of tensors (one tensor per feature: counts, page_ix, ..., count_variance)
        print('features',features)
    
    
#        for _ in range(10):#max(n_epoch,20)):
#            print('\n'*5)
#            random_draw_new_window_sizes()
#            print('max_train_empty',self.max_train_empty)
#            print('max_predict_empty',self.max_predict_empty)
#            print('history_window_size',self.history_window_size)
#            print('horizon_window_size',self.horizon_window_size)
#            print('attn_window',self.attn_window)            
#            
#            root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
#    #        print(root_ds.output_classes, root_ds.output_shapes, root_ds.output_types,)
#            print('root_ds.output_shapes',root_ds.output_shapes)
#            print('root_ds.output_types',root_ds.output_types)
#    #        batch = (root_ds
#    #                 .map(cutter[mode])
#    #                 .filter(self.reject_filter)
#    #                 .map(self.make_features, num_parallel_calls=num_threads)
#    #                 .batch(batch_size)
#    #                 .prefetch(runs_in_burst * 2)
#    #                 )
#            
#            #TEST:change horisoron jiostory
#            batch = root_ds.map(cutter[mode]).filter(self.reject_filter).map(self.make_features, num_parallel_calls=num_threads)
#            print('batch MFM', batch)
#            
#            batch = batch.batch(batch_size)
#            print('batch B', batch)
#             
#            batch = batch.prefetch(runs_in_burst * 2)
#            print('batch P', batch)
#            batch = (batch)

        root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
        batch = (root_ds
#                 .map(self.randomize_window_sizes)
                 .map(cutter[mode])
                 .filter(self.reject_filter)
                 .map(self.make_features, num_parallel_calls=num_threads)
                 .batch(batch_size)
                 .prefetch(runs_in_burst * 2)
                 )        
        
        print('---------------- Done batching ----------------')
        print(batch)
        self.iterator = batch.make_initializable_iterator()
        it_tensors = self.iterator.get_next()
#        print('self.iterator',self.iterator)
#        print('it_tensors',it_tensors)

        # Assign all tensors to class variables
        #self.time_x is the tensor of features, regardless of which feature set, so this can stay same.
        if self.features_set=='arturius':
            self.true_x, self.time_x, self.norm_x, self.lagged_x, self.true_y, self.time_y, self.norm_y, self.norm_mean, \
            self.norm_std, self.series_features, self.page_ix = it_tensors
        elif self.features_set=='full':
            self.true_x, self.time_x, self.norm_x, self.true_y, self.time_y, self.norm_y, self.norm_mean, \
            self.norm_std, self.series_features, self.page_ix = it_tensors
        print('self.true_x', self.true_x)
#        self.true_x = tf.Print(self.true_x,['self.true_x',self.true_x])
        print('self.time_x', self.time_x)
#        print('self.time_y', self.time_y)
#        self.time_y = tf.Print(self.time_y,['self.time_y',self.time_y])
        """if self.features_set=='simple':
            pass
#        if self.features_set=='full':
#            pass
        if self.features_set=='full_w_context':
            pass"""

        self.encoder_features_depth = self.time_x.shape[2].value
        print('self.encoder_features_depth',self.encoder_features_depth)
        print('self.time_x.shape',self.time_x.shape)
        
    def load_vars(self, session):
        self.inp.restore(session)

    def init_iterator(self, session):
        session.run(self.iterator.initializer)


def page_features(inp: VarFeeder, features_set):
    """
    Other than inp.counts, these features are the static features.
    So do not need to pass in here the time-varying ones like day of week, 
    month of year, lagged, etc.
    
    DO NOT return dow, woy, moy, year, doy, holidays
    """
    
    if features_set=='arturius':
        d = (inp.counts, inp.pf_agent, inp.pf_country, inp.pf_site,
                inp.page_ix, inp.count_median, inp.year_autocorr, inp.quarter_autocorr,
                inp.count_pctl_100
                )
        
#    elif features_set=='simple':
#        raise Exception('not ready yet')
        
    elif features_set=='full':
        d = (inp.counts,
            inp.page_ix,
            inp.count_median,
#            inp.year_autocorr, inp.quarter_autocorr,
            inp.count_pctl_0,
            inp.count_pctl_5,
            inp.count_pctl_25,
            inp.count_pctl_75,
            inp.count_pctl_95,
            inp.count_pctl_100,
            inp.count_variance)          
        
#    elif features_set=='full_w_context':
#        raise Exception('not ready yet')
    
    return d