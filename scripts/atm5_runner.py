import pandas as pd
import sys
import csv
import numpy as np
import pickle
from sklearn.metrics import average_precision_score
from runner import Runner
from util import Util

# 定数
shap_sampling = 10000
corr_sampling = 10000


class atm5_Runner(Runner):
    def __init__(self, run_name, model_cls, features, setting, params, cv):
        super().__init__(run_name, model_cls, features, setting, params, cv)
        self.metrics = average_precision_score
        self.categoricals = ['exc_wl']
        # 各fold,groupのスコアをファイルに出力するための2次元リスト
        self.score_list = []
        self.score_list.append(['run_name', self.run_name])
        self.folf_score_list = []
        self.chip_score_list = []
        self.chip_exc_wl_score_list = []

    def train_fold(self, i_fold):
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != 'all'
        train_x = self.train_x.copy()
        train_y = self.train_y.copy()

        if validation:

            # 学習データ・バリデーションデータのindexを取得
            if self.cv_method == 'KFold':
                tr_idx, va_idx = self.load_index_k_fold(i_fold)
            elif self.cv_method == 'StratifiedKFold':
                tr_idx, va_idx = self.load_index_sk_fold(i_fold)
            elif self.cv_method == 'GroupKFold':
                tr_idx, va_idx = self.load_index_gk_fold_shuffle(i_fold)
            elif self.cv_method == 'StratifiedGroupKFold':
                tr_idx, va_idx = self.load_index_sgk_fold(i_fold)
            elif self.cv_method == 'TrainTestSplit':
                tr_idx, va_idx = self.load_index_train_test_split()
            elif self.cv_method == 'CustomTimeSeriesSplitter':
                tr_idx, va_idx = self.load_index_custom_ts_fold(i_fold)
            else:
                print('CVメソッドが正しくないため終了します')
                sys.exit(0)

            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # =================================
            """
            # pseudo labelingデータを追加する
            pseudo_df = pd.read_pickle(self.feature_dir_name + 'pseudo_labeling_lgb_StratifiedKFold_06061143.pkl')
            pseudo_df_x = pseudo_df.drop('target', axis=1)[self.features]
            pseudo_df_y = pseudo_df['target']
            # 結合
            tr_x = pd.concat([tr_x, pseudo_df_x], axis=0)
            tr_y = pd.concat([tr_y, pseudo_df_y], axis=0)

            # 追加
            pseudo_df = pd.read_pickle(self.feature_dir_name + 'pseudo_labeling_cat_StratifiedKFold_06061151depth5_pl_lgb_StratifiedKFold_06061146.pkl')
            pseudo_df_x = pseudo_df.drop('target', axis=1)[self.features]
            pseudo_df_y = pseudo_df['target']
            # 結合
            tr_x = pd.concat([tr_x, pseudo_df_x], axis=0)
            tr_y = pd.concat([tr_y, pseudo_df_y], axis=0)
            """
            # =================================

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価を行う
            if self.calc_shap:
                va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
            else:
                va_pred = model.predict(va_x)

            score = self.metrics(va_y, va_pred)
            # print(f'mean:{va_pred.mean()}')
            # print(f'std:{va_pred.std()}')
            # print(f'max:{va_pred.max()}')
            # print(f'min:{va_pred.min()}')
            # Util.dump_df_pickle(pd.DataFrame(va_pred), self.out_dir_name + f'{self.run_name}_{i_fold}_vapred_df.pkl')
            # Util.dump_df_pickle(va_pred, self.out_dir_name + f'{self.run_name}_{i_fold}_vapred.pkl')
            va_pred_normalized = (va_pred - va_pred.mean()) / va_pred.std()
            va_pred_minmax = (va_pred - va_pred.mean()) / (va_pred.max() - va_pred.min())

            # =================================
            # 特別仕様: groupごとのスコアを算出
            _temp_df = pd.read_pickle(self.feature_dir_name + 'X_train.pkl')[['chip_id', 'chip_exc_wl']]
            _temp_df = _temp_df.iloc[va_idx].reset_index(drop=True)
            _temp_df = pd.concat([_temp_df, va_y.reset_index(drop=True), pd.Series(va_pred, name='pred')], axis=1)

            # chip_idの辞書
            with open(self.feature_dir_name + 'chip_dic.pkl', 'rb') as f:
                chip_dict = pickle.load(f)

            for i in sorted(_temp_df['chip_id'].unique().tolist()):
                chip_df = _temp_df.query('chip_id == @i')
                chip_y = chip_df['target']
                chip_pred = chip_df['pred']
                chip_score = self.metrics(chip_y, chip_pred)
                # chip_idごとのスコアをリストに追加
                self.chip_score_list.append([chip_dict[i], round(chip_score, 4)])

            for i in sorted(_temp_df['chip_exc_wl'].unique().tolist()):
                chip_exc_wl_df = _temp_df.query('chip_exc_wl == @i')
                chip_exc_wl_y = chip_exc_wl_df['target']
                chip_exc_wl_pred = chip_exc_wl_df['pred']
                chip_exc_wl_score = self.metrics(chip_exc_wl_y, chip_exc_wl_pred)
                # chip_exc_wlごとのスコアをリストに追加
                self.chip_exc_wl_score_list.append([i, round(chip_exc_wl_score, 4)])

            # foldごとのスコアをリストに追加
            self.folf_score_list.append([f'fold{i_fold}', round(score, 4)])
            # =================================

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, va_pred_minmax, va_pred_normalized, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        self.logger.info(f'{self.run_name} - start training cv')
        if self.cv_method in ['KFold', 'TrainTestSplit', 'CustomTimeSeriesSplitter']:
            self.logger.info(f'{self.run_name} - cv method: {self.cv_method}')
        else:
            self.logger.info(f'{self.run_name} - cv method: {self.cv_method} - group: {self.cv_target_gr_column} - stratify: {self.cv_target_sf_column}')

        scores = []  # 各foldのscoreを保存
        va_idxes = []  # 各foldのvalidationデータのindexを保存
        preds = []  # 各foldの推論結果を保存
        preds_minmax = []  # 各foldのminmax推論結果を保存
        preds_normalized = []  # 各foldのnormalize推論結果を保存

        # 各foldで学習を行う
        for i_fold in range(self.n_splits):
            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, va_pred_minmax, va_pred_normalized, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model(self.out_dir_name)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)
            preds_minmax.append(va_pred_minmax)
            preds_normalized.append(va_pred_normalized)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        preds_minmax = np.concatenate(preds_minmax, axis=0)
        preds_minmax = preds_minmax[order]
        preds_normalized = np.concatenate(preds_normalized, axis=0)
        preds_normalized = preds_normalized[order]

        # 全体のスコアを算出（これを平均とする）
        score_all_data = self.metrics(self.train_y, preds)

        # =================================
        # csvに書き込む
        self.score_list.append(['score_all_data', score_all_data])
        self.score_list.append(['score_fold_mean', np.mean(scores)])
        for i in self.folf_score_list:
            self.score_list.append(i)
        for i in self.chip_score_list:
            self.score_list.append(i)
        for i in self.chip_exc_wl_score_list:
            self.score_list.append(i)
        # with open(self.model_dir_name + self.out_dir_name + '/' + self.run_name + '_score.csv', 'a') as f:
        with open(self.out_dir_name + f'{self.run_name}_score.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(self.score_list)
        # =================================

        self.logger.info(f'{self.run_name} - end training cv - mean score:{np.mean(scores)} - alldata mean:{score_all_data}')

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(preds), self.out_dir_name + f'{self.run_name}_train.pkl')
            Util.dump_df_pickle(pd.DataFrame(preds_minmax), self.out_dir_name + f'{self.run_name}_train_minmax.pkl')
            Util.dump_df_pickle(pd.DataFrame(preds_normalized), self.out_dir_name + f'{self.run_name}_train_normalized.pkl')

        # 評価結果の保存
        self.logger.result_scores(self.run_name, scores, score_all_data)

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance()
