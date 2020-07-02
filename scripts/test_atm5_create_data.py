"""特徴量生成スクリプト
"""

import pandas as pd
import numpy as np
import yaml
import time
import pathlib
from functools import wraps
import category_encoders as ce
from tqdm import tqdm
from util import Logger
from slack_notify import slack_notify

tqdm.pandas()

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']
SLACK_NOTIFY = yml['SETTING']['SLACK_NOTIFY']


def elapsed_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info_log('Start: {}'.format(f.__name__))
        if SLACK_NOTIFY:
            slack_notify(notify_type='1', title='Start: {}'.format(f.__name__), value=None, run_name=None, ts=ts)
        v = f(*args, **kwargs)
        logger.info_log('End: {} done in {:.2f} s'.format(f.__name__, (time.time() - start)))
        if SLACK_NOTIFY:
            slack_notify(notify_type='1', title='End: {} done in {:.2f} s'
                         .format(f.__name__, (time.time() - start)), value=None, run_name=run_name, ts=ts)
        return v
    return wrapper


@elapsed_time
def load_data(create_pkl=False):
    """データの読み込み"""
    if create_pkl:
        # 基礎となるデータフレームをpklとして生成
        _train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
        _test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')
        _fitting = pd.read_csv(RAW_DATA_DIR_NAME + 'fitting.csv')
        _train = pd.merge(_train, _fitting, on="spectrum_id", how="left")
        _test = pd.merge(_test, _fitting, on="spectrum_id", how="left")

        p_temp = pathlib.Path(RAW_DATA_DIR_NAME + 'spectrum_raw')
        spec = []
        for file in p_temp.iterdir():
            spec_df = pd.read_csv(file, sep='\t', header=None)
            spec_df.columns = ["wavelength", "intensity"]
            spec_df["spectrum_filename"] = file.stem + ".dat"
            spec.append(spec_df)
        _train.to_pickle(RAW_DATA_DIR_NAME + 'train.pkl')
        _test.to_pickle(RAW_DATA_DIR_NAME + 'test.pkl')
        spec_df.to_pickle(RAW_DATA_DIR_NAME + 'spectrum.pkl')

    train = pd.read_pickle(RAW_DATA_DIR_NAME + 'train.pkl')
    test = pd.read_pickle(RAW_DATA_DIR_NAME + 'test.pkl')
    spectrum = pd.read_pickle(RAW_DATA_DIR_NAME + 'spectrum.pkl')

    return train, test, spectrum


@elapsed_time
def category_encode(df):
    categorical_col = [
        'chip_id',
        'exc_wl',
        'layout_a',
        'chip_exc_wl_label',
        'layout_type'
    ]

    # chip_idとexc_wlごとのスコアを見るためだけのカラムを生成
    df['chip_exc_wl_label'] = df['chip_id'].astype(str) + '_' + df['exc_wl'].astype(str)
    df['chip_exc_wl'] = df['chip_id'].astype(str) + '_' + df['exc_wl'].astype(str)
    df['chip'] = df['chip_id']

    ce_oe = ce.OrdinalEncoder(cols=categorical_col, handle_unknown='impute')

    # カテゴリ変換
    df[categorical_col] = ce_oe.fit_transform(df[categorical_col])
    df['chip_exc_wl_label'] = df['chip_exc_wl_label'] - 1  # chip_exc_wl_labelを目的変数として学習する際の設定（0~9に変換）

    return df


@elapsed_time
def add_category_agg(df):
    """カテゴリ変数ごとの集約特徴量とその差分"""

    col_float = []
    for p in np.arange(1, 0.48, -0.02):
        col_suffix = str(round(p, 2)).replace('.', '')
        col_float.append(f'fwhm_{col_suffix}')
        col_float.append(f'num_peak_{col_suffix}')

    agg_col = [
        'params0',
        'params1',
        'params2',
        'params3',
        'params4',
        'params5',
        'params6',
        'rms',
        'beta',
        'intensity_amplitude_v',
        'intensity_max',
        'intensity_min',
        'intensity_q90',
        'intensity_q75',
        # feature_importanceの高いもの
        'intensity__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
        'intensity__ar_coefficient__coeff_2__k_10',
        'intensity__fft_coefficient__attr_"abs"__coeff_65',
        'intensity__fft_coefficient__attr_"abs"__coeff_72',
        'params2_multi_params5',
        'fwhm_mult_num_peak_086',
        'intensity_max_divid_q80',
        'dc_umap2',
        'fwhm_mult_num_peak_088',
        'params3_divid_params1',
        'intensity__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
        'params5_multi_params2',
        'dc_tsne2',
        'fwhm_mult_num_peak_084'
    ] + col_float

    target_col = ['chip_id', 'exc_wl', 'layout_a', 'layout_type']

    for col in target_col:
        _df = df.groupby(col)[agg_col].mean().reset_index()
        _df = _df.add_prefix(f'{col}_mean_')
        df = pd.merge(df, _df, left_on=col, right_on=f'{col}_mean_{col}', how='left').drop(columns=f'{col}_mean_{col}')

        # 自身との差分を計算
        for i in agg_col:
            df[f'diff_{col}_mean_{i}'] = df[f'{col}_mean_{i}'] - df[i]

    return df


@elapsed_time
def feature_eng(train, test):
    """特徴量生成処理"""
    # 結合
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    # カテゴリ変換
    df = category_encode(df)

    # カテゴリ毎 * 数値変数の特徴量生成
    df = add_category_agg(df)

    return df.iloc[:len(train), :], df.iloc[len(train):, :]


@elapsed_time
def main(create_pkl):

    logger.info_log('★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆')

    # laod data
    train, test, spectrum = load_data(create_pkl)

    # 目的変数を切り出し
    y_train = train[['target']]
    train = train.drop('target', axis=1)

    # 特徴量生成
    X_train, X_test = feature_eng(train, test)

    logger.info_log('Save train test')
    train = pd.concat([X_train, y_train], axis=1)
    test = X_test
    train.to_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test.to_pickle(FEATURE_DIR_NAME + 'test.pkl')

    logger.info_log(f'train shape: {train.shape}, test shape, {test.shape}')

    return 'main() Done!'


if __name__ == "__main__":
    global logger, run_name, ts
    logger = Logger()

    if SLACK_NOTIFY:
        run_name = '前処理'
        res = slack_notify(notify_type='1', title='処理開始', value=None, run_name=run_name)
        ts = res['ts']

    # 全部の特徴量を作る場合はすべてTrueで実行する
    main(create_pkl=False)
