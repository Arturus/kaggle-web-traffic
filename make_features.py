import pandas as pd
import numpy as np
import os.path
import os
import argparse

import extractor
from feeder import VarFeeder
import numba
from typing import Tuple, Dict, Collection, List


def read_cached(name) -> pd.DataFrame:
    """
    Reads csv file (maybe zipped) from data directory and caches it's content as a pickled DataFrame
    :param name: file name without extension
    :return: file content
    """
    cached = 'data/%s.pkl' % name
    sources = ['data/%s.csv' % name, 'data/%s.csv.zip' % name]
    if os.path.exists(cached):
        return pd.read_pickle(cached)
    else:
        for src in sources:
            if os.path.exists(src):
                df = pd.read_csv(src)
                df.to_pickle(cached)
                return df


def read_all() -> pd.DataFrame:
    """
    Reads source data for training/prediction
    """
    def read_file(file):
        df = read_cached(file).set_index('Page')
        df.columns = df.columns.astype('M8[D]')
        return df

    # Path to cached data
    path = os.path.join('data', 'all.pkl')
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        # Official data
        df = read_file('train_2')
        # Scraped data
        scraped = read_file('2017-08-15_2017-09-11')
        # Update last two days by scraped data
        df[pd.Timestamp('2017-09-10')] = scraped['2017-09-10']
        df[pd.Timestamp('2017-09-11')] = scraped['2017-09-11']

        df = df.sort_index()
        # Cache result
        df.to_pickle(path)
    return df

# todo:remove
def make_holidays(tagged, start, end) -> pd.DataFrame:
    def read_df(lang):
        result = pd.read_pickle('data/holidays/%s.pkl' % lang)
        return result[~result.dw].resample('D').size().rename(lang)

    holidays = pd.DataFrame([read_df(lang) for lang in ['de', 'en', 'es', 'fr', 'ja', 'ru', 'zh']])
    holidays = holidays.loc[:, start:end].fillna(0)
    result =tagged[['country']].join(holidays, on='country').drop('country', axis=1).fillna(0).astype(np.int8)
    result.columns = pd.DatetimeIndex(result.columns.values)
    return result


def read_x(start, end) -> pd.DataFrame:
    """
    Gets source data from start to end date. Any date can be None
    """
    df = read_all()
    # User GoogleAnalitycsRoman has really bad data with huge traffic spikes in all incarnations.
    # Wikipedia banned him, we'll ban it too
    bad_roman = df.index.str.startswith("User:GoogleAnalitycsRoman")
    df = df[~bad_roman]
    if start and end:
        return df.loc[:, start:end]
    elif end:
        return df.loc[:, :end]
    else:
        return df


@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


@numba.jit(nopython=True)
def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag-1)
            c_366 = single_autocorr(series, lag+1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    return corr #, support


@numba.jit(nopython=True)
def find_start_end(data: np.ndarray):
    """
    Calculates start and end of real traffic data. Start is an index of first non-zero, non-NaN value,
     end is index of last non-zero, non-NaN value
    :param data: Time series, shape [n_pages, n_days]
    :return:
    """
    n_pages = data.shape[0]
    n_days = data.shape[1]
    start_idx = np.full(n_pages, -1, dtype=np.int32)
    end_idx = np.full(n_pages, -1, dtype=np.int32)
    for page in range(n_pages):
        # scan from start to the end
        for day in range(n_days):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                start_idx[page] = day
                break
        # reverse scan, from end to start
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                end_idx[page] = day
                break
    return start_idx, end_idx


def prepare_data(start, end, valid_threshold) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Reads source data, calculates start and end of each series, drops bad series, calculates log1p(series)
    :param start: start date of effective time interval, can be None to start from beginning
    :param end: end date of effective time interval, can be None to return all data
    :param valid_threshold: minimal ratio of series real length to entire (end-start) interval. Series dropped if
    ratio is less than threshold
    :return: tuple(log1p(series), nans, series start, series end)
    """
    df = read_x(start, end)
    starts, ends = find_start_end(df.values)
    # boolean mask for bad (too short) series
    page_mask = (ends - starts) / df.shape[1] < valid_threshold
    print("Masked %d pages from %d" % (page_mask.sum(), len(df)))
    inv_mask = ~page_mask
    df = df[inv_mask]
    nans = pd.isnull(df)
    return np.log1p(df.fillna(0)), nans, starts[inv_mask], ends[inv_mask]


def lag_indexes(begin, end) -> List[pd.Series]:
    """
    Calculates indexes for 3, 6, 9, 12 months backward lag for the given date range
    :param begin: start of date range
    :param end: end of date range
    :return: List of 4 Series, one for each lag. For each Series, index is date in range(begin, end), value is an index
     of target (lagged) date in a same Series. If target date is out of (begin,end) range, index is -1
    """
    dr = pd.date_range(begin, end)
    # key is date, value is day index
    base_index = pd.Series(np.arange(0, len(dr)), index=dr)

    def lag(offset):
        dates = dr - offset
        return pd.Series(data=base_index.loc[dates].fillna(-1).astype(np.int16).values, index=dr)

    return [lag(pd.DateOffset(months=m)) for m in (3, 6, 9, 12)]


def make_page_features(pages: np.ndarray) -> pd.DataFrame:
    """
    Calculates page features (site, country, agent, etc) from urls
    :param pages: Source urls
    :return: DataFrame with features as columns and urls as index
    """
    tagged = extractor.extract(pages).set_index('page')
    # Drop useless features
    features: pd.DataFrame = tagged.drop(['term', 'marker'], axis=1)
    return features


def uniq_page_map(pages:Collection):
    """
    Finds agent types (spider, desktop, mobile, all) for each unique url, i.e. groups pages by agents
    :param pages: all urls (must be presorted)
    :return: array[num_unique_urls, 4], where each column corresponds to agent type and each row corresponds to unique url.
     Value is an index of page in source pages array. If agent is missing, value is -1
    """
    import re
    result = np.full([len(pages), 4], -1, dtype=np.int32)
    pat = re.compile(
        '(.+(?:(?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.mediawiki\.org)))_([a-z_-]+?)')
    prev_page = None
    num_page = -1
    agents = {'all-access_spider': 0, 'desktop_all-agents': 1, 'mobile-web_all-agents': 2, 'all-access_all-agents': 3}
    for i, entity in enumerate(pages):
        match = pat.fullmatch(entity)
        assert match
        page = match.group(1)
        agent = match.group(2)
        if page != prev_page:
            prev_page = page
            num_page += 1
        result[num_page, agents[agent]] = i
    return result[:num_page+1]


def encode_page_features(df) -> Dict[str, pd.DataFrame]:
    """
    Applies one-hot encoding to page features and normalises result
    :param df: page features DataFrame (one column per feature)
    :return: dictionary feature_name:encoded_values. Encoded values is [n_pages,n_values] array
    """
    def encode(column) -> pd.DataFrame:
        one_hot = pd.get_dummies(df[column], drop_first=False)
        # noinspection PyUnresolvedReferences
        return (one_hot - one_hot.mean()) / one_hot.std()

    return {str(column): encode(column) for column in df}


def normalize(values: np.ndarray):
    return (values - values.mean()) / np.std(values)


def run():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('data_dir')
    parser.add_argument('--valid_threshold', default=0.0, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--add_days', default=64, type=int, help="Add N days in a future for prediction")
    parser.add_argument('--start', help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', help="Effective end date. Data past the end is dropped")
    parser.add_argument('--corr_backoffset', default=0, type=int, help='Offset for correlation calculation')
    args = parser.parse_args()

    # Get the data
    df, nans, starts, ends = prepare_data(args.start, args.end, args.valid_threshold)

    # Our working date range
    data_start, data_end = df.columns[0], df.columns[-1]

    # We have to project some date-dependent features (day of week, etc) to the future dates for prediction
    features_end = data_end + pd.Timedelta(args.add_days, unit='D')
    print(f"start: {data_start}, end:{data_end}, features_end:{features_end}")

    # Group unique pages by agents
    assert df.index.is_monotonic_increasing
    page_map = uniq_page_map(df.index.values)

    # Yearly(annual) autocorrelation
    raw_year_autocorr = batch_autocorr(df.values, 365, starts, ends, 1.5, args.corr_backoffset)
    year_unknown_pct = np.sum(np.isnan(raw_year_autocorr))/len(raw_year_autocorr)  # type: float

    # Quarterly autocorrelation
    raw_quarter_autocorr = batch_autocorr(df.values, int(round(365.25/4)), starts, ends, 2, args.corr_backoffset)
    quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr)  # type: float

    print("Percent of undefined autocorr = yearly:%.3f, quarterly:%.3f" % (year_unknown_pct, quarter_unknown_pct))

    # Normalise all the things
    year_autocorr = normalize(np.nan_to_num(raw_year_autocorr))
    quarter_autocorr = normalize(np.nan_to_num(raw_quarter_autocorr))

    # Calculate and encode page features
    page_features = make_page_features(df.index.values)
    encoded_page_features = encode_page_features(page_features)

    # Make time-dependent features
    features_days = pd.date_range(data_start, features_end)
    #dow = normalize(features_days.dayofweek.values)
    week_period = 7 / (2 * np.pi)
    dow_norm = features_days.dayofweek.values / week_period
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)

    # Assemble indices for quarterly lagged data
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)

    page_popularity = df.median(axis=1)
    page_popularity = (page_popularity - page_popularity.mean()) / page_popularity.std()

    # Put NaNs back
    df[nans] = np.NaN

    # Assemble final output
    tensors = dict(
        hits=df,
        lagged_ix=lagged_ix,
        page_map=page_map,
        page_ix=df.index.values,
        pf_agent=encoded_page_features['agent'],
        pf_country=encoded_page_features['country'],
        pf_site=encoded_page_features['site'],
        page_popularity=page_popularity,
        year_autocorr=year_autocorr,
        quarter_autocorr=quarter_autocorr,
        dow=dow,
    )
    plain = dict(
        features_days=len(features_days),
        data_days=len(df.columns),
        n_pages=len(df),
        data_start=data_start,
        data_end=data_end,
        features_end=features_end

    )

    # Store data to the disk
    VarFeeder(args.data_dir, tensors, plain)


if __name__ == '__main__':
    run()
