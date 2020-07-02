"""atmaCup#5の特徴量生成スクリプト
"""

import pandas as pd
import numpy as np
import math
import yaml
import time
import pickle
import pathlib
from scipy.interpolate import UnivariateSpline
from functools import wraps
import category_encoders as ce
from tsfresh import extract_features, extract_relevant_features
import umap
import bhtsne
from tqdm import tqdm

from util import Logger

tqdm.pandas()

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']


def elapsed_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info_log('Start: {}'.format(f.__name__))
        # slack_notify(notify_type='1', title='Start: {}'.format(f.__name__), value=None, run_name=run_name, ts=ts)
        v = f(*args, **kwargs)
        logger.info_log('End: {} done in {:.2f} s'.format(f.__name__, (time.time() - start)))
        # slack_notify(notify_type='1', title='End: {} done in {:.2f} s'\
        # .format(f.__name__, (time.time() - start)), value=None, run_name=run_name, ts=ts)
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
def calculate_peak_fwhm(df) -> pd.DataFrame:
    """値幅とピークを算出
        1時間ほどかかる
    """
    spectrum_filename_list = df['spectrum_filename'].unique().tolist()
    spectrum_dict = {}
    for file_name in tqdm(spectrum_filename_list):
        x = df.query('spectrum_filename == @file_name')['wavelength'].values
        y = df.query('spectrum_filename == @file_name')['intensity'].values
        # 2%刻みの値幅を計算
        col_float = []  # 保存するカラムを保持
        temp_dict = {}
        for p in np.arange(1, 0.48, -0.02):
            spline = UnivariateSpline(x, y-np.max(y)*round(p, 2), s=0)
            roots = spline.roots()  # find the roots

            # 値幅
            fwhm = np.nan
            for i in range(len(roots)//2):
                temp_fwhm = roots[i*2+1] - roots[i*2]
                if i == 0:
                    fwhm = temp_fwhm
                else:
                    if fwhm > temp_fwhm:
                        fwhm = temp_fwhm
            # ピークの回数
            num_peak = len(roots)//2

            col_suffix = str(round(p, 2)).replace('.', '')
            temp_dict[f'fwhm_{col_suffix}'] = fwhm
            temp_dict[f'num_peak_{col_suffix}'] = num_peak
            col_float.append(f'fwhm_{col_suffix}')
            col_float.append(f'num_peak_{col_suffix}')

        spectrum_dict[file_name] = temp_dict
    temp_df = pd.DataFrame.from_dict(spectrum_dict, orient='index')\
                .reset_index().rename({'index': 'spectrum_filename'}, axis=1)

    col = ['spectrum_filename'] + col_float
    temp_df[col].to_pickle(FEATURE_DIR_NAME + 'spectrum_peak_fwhm_df.pkl')
    return temp_df


@elapsed_time
def calculate_tsfresh_df(train, spectrum):
    """tsfreshを用いて大量の特徴量を生成する
        処理に1時間ほど要する
    """

    # 特徴量生成
    X = extract_features(spectrum, column_id="spectrum_filename", column_sort="wavelength")

    # 重要な特徴量に絞る
    y = pd.merge(train, spectrum, how="left", on="spectrum_filename").set_index("spectrum_filename").target
    y = y.groupby("spectrum_filename").mean()
    # trainに含まれるスペクトルファイル
    train_spectrum_filename_list = train['spectrum_filename'].unique().tolist()
    train_spectrum = spectrum[spectrum['spectrum_filename'].isin(train_spectrum_filename_list)]
    X2 = extract_relevant_features(train_spectrum, y, column_id="spectrum_filename", column_sort="wavelength")

    cols = X2.columns.tolist()
    tsfresh_df = X[cols].reset_index()
    tsfresh_df.rename(columns={'id': 'spectrum_filename'}, inplace=True)
    tsfresh_df.to_pickle(FEATURE_DIR_NAME + 'tsfresh_df.pkl')


@elapsed_time
def fe_spectrum(df, calculate_fwhm=False, calculate_tsneumap=False) -> pd.DataFrame:
    """スペクトルの特徴量生成処理"""

    # 光の強度の集計統計量
    _df = df.groupby('spectrum_filename')['intensity']\
        .agg(['max', 'min', 'mean', 'std', 'sum', 'median'])\
        .rename(columns={'max': 'intensity_max', 'min': 'intensity_min',
                         'mean': 'intensity_mean', 'std': 'intensity_std',
                         'sum': 'intensity_sum', 'median': 'intensity_median'})

    # 縦の振幅の大きさ
    _df['intensity_amplitude_v'] = _df['intensity_max'] - _df['intensity_min']

    # quantile
    # 10%, 25%, 50%, 75%, 90%
    q10 = df.groupby('spectrum_filename')['intensity'].quantile(0.1).reset_index()\
        .rename({'intensity': 'intensity_q10'}, axis=1)
    q25 = df.groupby('spectrum_filename')['intensity'].quantile(0.25).reset_index()\
        .rename({'intensity': 'intensity_q25'}, axis=1)
    q50 = df.groupby('spectrum_filename')['intensity'].quantile(0.5).reset_index()\
        .rename({'intensity': 'intensity_q50'}, axis=1)
    q75 = df.groupby('spectrum_filename')['intensity'].quantile(0.75).reset_index()\
        .rename({'intensity': 'intensity_q75'}, axis=1)
    q80 = df.groupby('spectrum_filename')['intensity'].quantile(0.8).reset_index()\
        .rename({'intensity': 'intensity_q80'}, axis=1)
    q85 = df.groupby('spectrum_filename')['intensity'].quantile(0.85).reset_index()\
        .rename({'intensity': 'intensity_q85'}, axis=1)
    q90 = df.groupby('spectrum_filename')['intensity'].quantile(0.9).reset_index()\
        .rename({'intensity': 'intensity_q90'}, axis=1)
    _df = pd.merge(_df, q10, how='left', on='spectrum_filename')
    _df = pd.merge(_df, q25, how='left', on='spectrum_filename')
    _df = pd.merge(_df, q50, how='left', on='spectrum_filename')
    _df = pd.merge(_df, q75, how='left', on='spectrum_filename')
    _df = pd.merge(_df, q80, how='left', on='spectrum_filename')
    _df = pd.merge(_df, q85, how='left', on='spectrum_filename')
    _df = pd.merge(_df, q90, how='left', on='spectrum_filename')

    # maxとパーセンタイル点の差分
    _df['intensity_max_minus_q90'] = _df['intensity_max'] - _df['intensity_q90']
    _df['intensity_max_minus_q85'] = _df['intensity_max'] - _df['intensity_q85']
    _df['intensity_max_minus_q80'] = _df['intensity_max'] - _df['intensity_q80']
    _df['intensity_max_minus_q75'] = _df['intensity_max'] - _df['intensity_q75']
    _df['intensity_max_minus_q50'] = _df['intensity_max'] - _df['intensity_q50']
    # maxとパーセンタイル点の積
    _df['intensity_max_multi_q90'] = _df['intensity_max'] * _df['intensity_q90']
    _df['intensity_max_multi_q85'] = _df['intensity_max'] * _df['intensity_q85']
    _df['intensity_max_multi_q80'] = _df['intensity_max'] * _df['intensity_q80']
    _df['intensity_max_multi_q75'] = _df['intensity_max'] * _df['intensity_q75']
    _df['intensity_max_multi_q50'] = _df['intensity_max'] * _df['intensity_q50']
    # max / パーセンタイル点
    _df['intensity_max_divid_q90'] = _df['intensity_max'] / _df['intensity_q90']
    _df['intensity_max_divid_q85'] = _df['intensity_max'] / _df['intensity_q85']
    _df['intensity_max_divid_q80'] = _df['intensity_max'] / _df['intensity_q80']
    _df['intensity_max_divid_q75'] = _df['intensity_max'] / _df['intensity_q75']
    _df['intensity_max_divid_q50'] = _df['intensity_max'] / _df['intensity_q50']
    # 90%タイル点と各タイル点との差分
    _df['intensity_q85_divid_q90'] = _df['intensity_q85'] / _df['intensity_q90']
    _df['intensity_q80_divid_q90'] = _df['intensity_q80'] / _df['intensity_q90']
    _df['intensity_q75_divid_q90'] = _df['intensity_q75'] / _df['intensity_q90']
    _df['intensity_q50_divid_q90'] = _df['intensity_q50'] / _df['intensity_q90']

    # 半値幅とピークの回数のデータを計算する
    # cf: https://ja.wikipedia.org/wiki/%E5%8D%8A%E5%80%A4%E5%B9%85
    # cf: https://www.guruguru.ml/competitions/10/discussions/1e745a02-d3d1-4be3-8043-25c37ae85dbd
    if calculate_fwhm:
        # 1時間ほどかかる
        temp_df = calculate_peak_fwhm(df)
    else:
        temp_df = pd.read_pickle(FEATURE_DIR_NAME + 'spectrum_peak_fwhm_df.pkl')
    _df = pd.merge(_df, temp_df, how='left', on='spectrum_filename')

    # 1ピークあたりのfwhmとピーク数*fwhmを計算
    for p in np.arange(1, 0.48, -0.02):
        col_suffix = str(round(p, 2)).replace('.', '')
        _df[f'fwhm_divid_num_peak_{col_suffix}'] = _df[f'fwhm_{col_suffix}'] / _df[f'num_peak_{col_suffix}']
        _df[f'fwhm_mult_num_peak_{col_suffix}'] = _df[f'fwhm_{col_suffix}'] * _df[f'num_peak_{col_suffix}']

    # TODO: フーリエ変換
    # cf:https://helve-python.hatenablog.jp/entry/2018/06/17/000000

    # t-SNE, UMAPで2次元に射影したデータ
    if calculate_tsneumap:
        temp = df.reset_index().pivot(index='spectrum_filename', columns='index', values='intensity')
        temp = temp.fillna(0)
        v_columns = [i for i in range(512)]
        # UMAP
        um = umap.UMAP(random_state=42)
        _umap = um.fit_transform(temp[v_columns])
        umap_df = pd.DataFrame(_umap, columns=['dc_umap1', 'dc_umap2'])

        # t-SNE
        _bhtsne = bhtsne.tsne(temp[v_columns].astype(np.float64), dimensions=2, rand_seed=42)
        bhtsne_df = pd.DataFrame(_bhtsne, columns=['dc_tsne1', 'dc_tsne2'])

        temp = temp.reset_index()[['spectrum_filename']]
        _df = pd.concat([temp, umap_df, bhtsne_df], axis=1)

        _df.to_pickle(FEATURE_DIR_NAME + 'wave_umap_tsne.pkl')

    wave_umap_tsne = pd.read_pickle(FEATURE_DIR_NAME + 'wave_umap_tsne.pkl')
    _df = pd.merge(_df, wave_umap_tsne, on='spectrum_filename', how='left')

    return _df.reset_index(drop=True)


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

    # chip_idのdictを生成（学習時にchipごとのスコアをみるため）
    chip_df = df[['chip_id']].copy()
    chip_df['chip_id_enc'] = ce_oe.fit_transform(df[categorical_col])[['chip_id']]
    # 重複を除外
    chip_df = chip_df[~chip_df.duplicated()].reset_index(drop=True)
    chip_dic = dict(zip(chip_df['chip_id_enc'], chip_df['chip_id']))
    with open(FEATURE_DIR_NAME + 'chip_dic.pkl', mode='wb') as f:
        pickle.dump(chip_dic, f)

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
def calculate_cavity_save(df, spectrum):
    """spectrumからcavityの特徴量を抽出
        30分ほどかかる
    """
    _temp = pd.merge(spectrum, df[['spectrum_filename', 'params0', 'params1', 'params2', 'params3',
                                   'params4', 'params5', 'params6']], on='spectrum_filename', how='left')

    target_df = pd.DataFrame()
    for filename in tqdm(_temp['spectrum_filename'].unique().tolist()):
        file_df = _temp.query('spectrum_filename == @filename')
        # params2の前後10の範囲の特徴量を生成する
        range_upper = file_df.loc[:, 'params2'].values[0] + 10
        range_lower = file_df.loc[:, 'params2'].values[0] - 10
        file_df = file_df.query('wavelength <= @range_upper & wavelength >= @range_lower')
        _df = file_df.groupby('spectrum_filename')['intensity']\
            .agg(['mean', 'std', 'median', 'skew'])\
            .rename(columns={'mean': 'cavity_mean', 'std': 'cavity_std',
                             'median': 'cavity_median', 'skew': 'cavity_skew'}).reset_index()
        _df['cavity_kurt'] = file_df['intensity'].kurt()
        target_df = target_df.append(_df)

    cavity_df = pd.merge(df[['spectrum_filename']], target_df, on='spectrum_filename', how='left')
    cavity_df.to_pickle(FEATURE_DIR_NAME + 'cavity_df.pkl')


@elapsed_time
def feature_eng(train, test, spectrum, calculate_cavity):
    """特徴量生成処理"""
    # 結合
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    # layoutx, layoutyの区画を設定
    def get_layout(row):
        x = row.layout_x
        y = row.layout_y
        if (0 <= x <= 24) & (0 <= y <= 96):
            return 1
        elif (25 <= x <= 47) & (0 <= y <= 96):
            return 2
        elif (0 <= x <= 24) & (97 <= y <= 191):
            return 3
        else:
            return 4
    df['layout_xy'] = df[['layout_x', 'layout_y']].apply(get_layout, axis=1)
    df['layout_type'] = df['layout_a'].astype(str) + '_' + df['layout_xy'].astype(str)

    # spectrumからcavity（params2）の特徴量を抽出
    if calculate_cavity:
        # 30分ほどかかる
        calculate_cavity_save(df, spectrum)
        cavity_df = pd.read_pickle(FEATURE_DIR_NAME + 'cavity_df.pkl')
    else:
        cavity_df = pd.read_pickle(FEATURE_DIR_NAME + 'cavity_df.pkl')
    df = pd.merge(df, cavity_df, on='spectrum_filename', how='left')

    # カテゴリ変換
    df = category_encode(df)

    # paramsの交互作用特徴量
    for x in range(7):
        # rmsとbeta
        df[f'params{x}_multi_rms'] = df[f'params{x}'] * df['rms']
        df[f'params{x}_divid_rms'] = df[f'params{x}'] / df['rms']
        df[f'params{x}_plus_rms'] = df[f'params{x}'] + df['rms']
        df[f'params{x}_minus_rms'] = df[f'params{x}'] - df['rms']
        df[f'params{x}_multi_beta'] = df[f'params{x}'] * df['beta']
        df[f'params{x}_divid_beta'] = df[f'params{x}'] / df['beta']
        df[f'params{x}_plus_beta'] = df[f'params{x}'] + df['beta']
        df[f'params{x}_minus_beta'] = df[f'params{x}'] - df['beta']

        for y in range(7):
            if x != y:
                # param同士
                df[f'params{x}_multi_params{y}'] = df[f'params{x}'] * df[f'params{y}']
                df[f'params{x}_divid_params{y}'] = df[f'params{x}'] / df[f'params{y}']
                df[f'params{x}_plus_params{y}'] = df[f'params{x}'] + df[f'params{y}']
                df[f'params{x}_minus_params{y}'] = df[f'params{x}'] - df[f'params{y}']
                df[f'params{x}_absminus_params{y}'] = abs(df[f'params{x}'] - df[f'params{y}'])

    # チップエリアの中心からの距離
    def get_distance(row):
        x1 = row['layout_x']
        y1 = row['layout_y']
        x2 = 24
        y2 = 96
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return d
    df.loc[:, 'distance'] = df.apply(get_distance, axis=1)
    # x, yそれぞれの中心からの距離
    df.loc[:, 'distance_x'] = abs(24 - df['layout_x'])
    df.loc[:, 'distance_y'] = abs(96 - df['layout_y'])

    # 事前に生成したtsfresh特徴量の結合
    tsfresh_df = pd.read_pickle(FEATURE_DIR_NAME + 'tsfresh_df.pkl')
    df = pd.merge(df, tsfresh_df, on='spectrum_filename', how='left')

    # camaroさんの特徴量をマージ
    # camaro_df = pd.read_pickle(FEATURE_DIR_NAME + 'camaro_spectrum.pkl')
    # df = pd.merge(df, camaro_df, on='spectrum_filename', how='left')

    # カテゴリ毎 * 数値変数の特徴量生成
    df = add_category_agg(df)

    return df.iloc[:len(train), :], df.iloc[len(train):, :]


@elapsed_time
def main(create_pkl, calculate_fwhm, calculate_tsfresh, calculate_cavity, calculate_tsneumap):

    logger.info_log('★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆')

    # laod data
    train, test, spectrum = load_data(create_pkl)

    # spectrumの特徴量取得
    spectrum_df = fe_spectrum(spectrum, calculate_fwhm, calculate_tsneumap)

    # tsfreshの特徴量を生成
    if calculate_tsfresh:
        # 1時間ほどかかる
        calculate_tsfresh_df(train, spectrum)

    # マージ
    train = pd.merge(train, spectrum_df, on='spectrum_filename', how='left')
    test = pd.merge(test, spectrum_df, on='spectrum_filename', how='left')

    # 目的変数を切り出し
    y_train = train[['target']]
    train = train.drop('target', axis=1)

    # 特徴量生成
    X_train, X_test = feature_eng(train, test, spectrum, calculate_cavity)

    # chip_idを予測対象とする場合に付与
    # y_train = X_train[['chip_exc_wl_label']]

    logger.info_log('Save train test')
    train = pd.concat([X_train, y_train], axis=1)
    test = X_test
    # X_train.to_pickle(FEATURE_DIR_NAME + 'X_train.pkl')
    # y_train.to_pickle(FEATURE_DIR_NAME + 'y_train.pkl')
    # X_test.to_pickle(FEATURE_DIR_NAME + 'X_test.pkl')
    train.to_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test.to_pickle(FEATURE_DIR_NAME + 'test.pkl')

    # logger.info_log(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape, {X_test.shape}')
    logger.info_log(f'train shape: {train.shape}, test shape, {test.shape}')

    return 'main() Done!'


if __name__ == "__main__":
    global logger
    logger = Logger()

    # 全部の特徴量を作る場合はすべてTrueで実行する
    main(create_pkl=False, calculate_fwhm=False, calculate_tsfresh=False, calculate_cavity=False, calculate_tsneumap=False)
