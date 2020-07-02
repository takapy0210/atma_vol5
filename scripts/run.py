import sys
import os
import datetime
import yaml
import json
import collections as cl
import warnings
import fire
import traceback
from atm5_model_lgb import atm5_ModelLGB
from model_cb import ModelCB
from atm5_runner import atm5_Runner
from util import Submission

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

key_list = ['use_features', 'model_params', 'cv', 'setting']

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']


def exist_check(path, run_name) -> None:
    """モデルディレクトリの存在チェック

    Args:
        path (str): モデルディレクトリのpath
        run_name (str): チェックするrun名

    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)
    return None


def my_makedirs(path) -> None:
    """引数のpathディレクトリが存在しなければ、新規で作成する

    Args:
        path (str): 作成するディレクトリ名

    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return None


def save_model_config(key_list, value_list, dir_name, run_name) -> None:
    """学習のjsonファイル生成

    どんなパラメータ/特徴量で学習させたモデルかを管理するjsonファイルを出力する

    """
    def set_default(obj):
        """json出力の際にset型のオブジェクトをリストに変更する"""
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    ys = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        ys[v] = data
    fw = open(dir_name + run_name + '_param.json', 'w')
    json.dump(ys, fw, indent=4, default=set_default)
    return None


def get_cv_info() -> dict:
    """CVの情報を設定する

    methodは[KFold, StratifiedKFold ,GroupKFold, StratifiedGroupKFold, CustomTimeSeriesSplitter, TrainTestSplit]から選択可能
    CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    StratifiedKFold or GroupKFold or StratifiedGroupKFold の場合はcv_target_gr, cv_target_sfに対象カラム名を設定する

    Returns:
        dict: cvの辞書

    """
    return yml['SETTING']['CV']


def get_run_name(cv, model_type):
    """run名を設定する
    """
    run_name = model_type
    suffix = '_' + datetime.datetime.now().strftime("%m%d%H%M")
    model_info = ''
    run_name = run_name + '_' + cv.get('method') + model_info + suffix
    return run_name


def get_setting_info():
    """setting情報を設定する
    """
    setting = {
        'feature_directory': yml['SETTING']['FEATURE_DIR_NAME'],  # 特徴量の読み込み先ディレクトリ
        'model_directory': MODEL_DIR_NAME,  # モデルの保存先ディレクトリ
        'train_file_name': yml['SETTING']['TRAIN_FILE_NAME'],
        'test_file_name': yml['SETTING']['TEST_FILE_NAME'],
        'target': yml['SETTING']['TARGET_COL'],  # 目的変数
        'calc_shap': yml['SETTING']['CALC_SHAP'],  # shap値を計算するか否か
        'save_train_pred': yml['SETTING']['SAVE_TRAIN_PRED']  # trainデータでの推論値を特徴量として加えたい場合はTrueに設定する
    }
    return setting


def main(model_type='lgb') -> str:
    """トレーニングのmain関数

    model_typeによって学習するモデルを変更する
    → lgb, cb, xgb, nnが標準で用意されている

    Args:
        model_type (str, optional): どのモデルで学習させるかを指定. Defaults to 'lgb'.

    Returns:
        str: [description]

    Examples:
        >>> python hoge.py --model_type="lgb"
        >>> python hoge.py lgb

    """

    cv = get_cv_info()  # CVの情報辞書
    run_name = get_run_name(cv, model_type)  # run名
    dir_name = MODEL_DIR_NAME + run_name + '/'  # 学習に使用するディレクトリ
    setting = get_setting_info()  # 諸々の設定ファイル辞書

    # すでに実行済みのrun名がないかチェックし、ディレクトリを作成する
    exist_check(MODEL_DIR_NAME, run_name)
    my_makedirs(dir_name)

    # モデルに合わせてパラメータを読み込む
    model_cls = None
    if model_type == 'lgb':
        model_params = yml['MODEL_LGB']['PARAM']
        model_cls = atm5_ModelLGB
    elif model_type == 'cb':
        model_params = yml['MODEL_CB']['PARAM']
        model_cls = ModelCB
    elif model_type == 'xgb':
        pass
    elif model_type == 'nn':
        pass
    else:
        print('model_typeが不正なため終了します')
        sys.exit(0)

    features = [
        'exc_wl',
        'distance',
        'distance_x',
        'distance_y',
        'pos_x',
        'params0',
        'params1',
        'params2',
        'params3',
        'params4',
        'params5',
        'params6',
        'params0_multi_rms',
        'params0_divid_rms',
        'params0_multi_beta',
        'params0_divid_beta',
        'params1_multi_rms',
        'params1_divid_rms',
        'params1_multi_beta',
        'params1_divid_beta',
        'params2_multi_rms',
        'params2_divid_rms',
        'params2_multi_beta',
        'params2_divid_beta',
        'params3_multi_rms',
        'params3_divid_rms',
        'params3_multi_beta',
        'params3_divid_beta',
        'params4_multi_rms',
        'params4_divid_rms',
        'params4_multi_beta',
        'params4_divid_beta',
        'params5_multi_rms',
        'params5_divid_rms',
        'params5_multi_beta',
        'params5_divid_beta',
        'params6_multi_rms',
        'params6_divid_rms',
        'params6_multi_beta',
        'params6_divid_beta',
        'params0_multi_params1',
        'params0_divid_params1',
        'params0_multi_params2',
        'params0_divid_params2',
        'params0_multi_params3',
        'params0_divid_params3',
        'params0_multi_params4',
        'params0_divid_params4',
        'params0_multi_params5',
        'params0_divid_params5',
        'params0_multi_params6',
        'params0_divid_params6',
        'params1_multi_params0',
        'params1_divid_params0',
        'params1_multi_params2',
        'params1_divid_params2',
        'params1_multi_params3',
        'params1_divid_params3',
        'params1_multi_params4',
        'params1_divid_params4',
        'params1_multi_params5',
        'params1_divid_params5',
        'params1_multi_params6',
        'params1_divid_params6',
        'params2_multi_params0',
        'params2_divid_params0',
        'params2_multi_params1',
        'params2_divid_params1',
        'params2_multi_params3',
        'params2_divid_params3',
        'params2_multi_params4',
        'params2_divid_params4',
        'params2_multi_params5',
        'params2_divid_params5',
        'params2_multi_params6',
        'params2_divid_params6',
        'params3_multi_params0',
        'params3_divid_params0',
        'params3_multi_params1',
        'params3_divid_params1',
        'params3_multi_params2',
        'params3_divid_params2',
        'params3_multi_params4',
        'params3_divid_params4',
        'params3_multi_params5',
        'params3_divid_params5',
        'params3_multi_params6',
        'params3_divid_params6',
        'params4_multi_params0',
        'params4_divid_params0',
        'params4_multi_params1',
        'params4_divid_params1',
        'params4_multi_params2',
        'params4_divid_params2',
        'params4_multi_params3',
        'params4_divid_params3',
        'params4_multi_params5',
        'params4_divid_params5',
        'params4_multi_params6',
        'params4_divid_params6',
        'params5_multi_params0',
        'params5_divid_params0',
        'params5_multi_params1',
        'params5_divid_params1',
        'params5_multi_params2',
        'params5_divid_params2',
        'params5_multi_params3',
        'params5_divid_params3',
        'params5_multi_params4',
        'params5_divid_params4',
        'params5_multi_params6',
        'params5_divid_params6',
        'params6_multi_params0',
        'params6_divid_params0',
        'params6_multi_params1',
        'params6_divid_params1',
        'params6_multi_params2',
        'params6_divid_params2',
        'params6_multi_params3',
        'params6_divid_params3',
        'params6_multi_params4',
        'params6_divid_params4',
        'params6_multi_params5',
        'params6_divid_params5',
        'rms',
        'beta',
        # --- TODO: スペクトル波長データ
        'intensity_max',
        'intensity_min',
        'intensity_mean',
        'intensity_std',
        'intensity_sum',
        'intensity_median',
        'intensity_amplitude_v',
        'intensity_q10',
        'intensity_q25',
        'intensity_q50',
        'intensity_q75',
        'intensity_q80',
        'intensity_q85',
        'intensity_q90',
        'intensity_max_minus_q90',
        'intensity_max_minus_q85',
        'intensity_max_minus_q80',
        'intensity_max_minus_q75',
        'intensity_max_minus_q50',
        'intensity_max_multi_q90',
        'intensity_max_multi_q85',
        'intensity_max_multi_q80',
        'intensity_max_multi_q75',
        'intensity_max_multi_q50',
        'intensity_max_divid_q90',
        'intensity_max_divid_q85',
        'intensity_max_divid_q80',
        'intensity_max_divid_q75',
        'intensity_max_divid_q50',
        'intensity_q85_divid_q90',
        'intensity_q80_divid_q90',
        'intensity_q75_divid_q90',
        'intensity_q50_divid_q90',
        # --- TODO: 波形の圧縮データ
        'dc_umap1',
        'dc_umap2',
        'dc_tsne1',
        'dc_tsne2',
        'diff_exc_wl_mean_params0',
        'diff_exc_wl_mean_params1',
        'diff_exc_wl_mean_params2',
        'diff_exc_wl_mean_params3',
        'diff_exc_wl_mean_params4',
        'diff_exc_wl_mean_params5',
        'diff_exc_wl_mean_params6',
        'diff_exc_wl_mean_rms',
        'diff_exc_wl_mean_beta',
        'diff_exc_wl_mean_intensity_amplitude_v',
        'diff_exc_wl_mean_fwhm_09',
    ]

    try:
        # インスタンス生成
        runner = atm5_Runner(run_name, model_cls, features, setting, model_params, cv)
        use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

        # モデルのconfigをjsonで保存
        value_list = [use_feature_name, model_params, cv, setting]
        save_model_config(key_list, value_list, dir_name, run_name)

        # 学習・推論
        runner.run_train_cv()
        runner.run_predict_cv()

        # submit作成
        Submission.create_submission(run_name, dir_name, setting.get('target'))

        if model_type == 'lgb':
            # feature_importanceを計算
            atm5_ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name, cv.get('n_splits'), type='gain')

    except Exception as e:
        print(traceback.format_exc())
        print(f'ERROR:{e}')


if __name__ == '__main__':
    fire.Fire(main)
