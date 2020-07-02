import pandas as pd
import sys
# import os
import csv
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from runner import Runner
# from model import Model
from util import Util

# 定数
shap_sampling = 10000
corr_sampling = 10000


class atm5_Runner(Runner):
    def __init__(self, run_name, model_cls, features, setting, params, cv, feature_dir_name, model_dir_name):

        super().__init__(run_name, model_cls, features, setting, params, cv, feature_dir_name, model_dir_name)
        # self.metrics = average_precision_score
        self.metrics = accuracy_score
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
            pseudo_df = pd.read_pickle(self.feature_dir_name + 'pseudo_labeling_camaro031_takapy_lgb_0601_2333.pkl')
            pseudo_df_x = pseudo_df.drop('target', axis=1)[self.features]
            pseudo_df_y = pseudo_df['target']
            # print(f'pseudo_df_x: {pseudo_df_x.shape}')
            # print(f'pseudo_df_y: {pseudo_df_y.shape}')
            # print(f'tr_x: {tr_x.shape}')
            # print(f'tr_y: {tr_y.shape}')
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

            # chip_idを予測する場合
            va_pred_prob = va_pred
            va_pred = np.argmax(va_pred, axis=1)  # 最尤と判断したクラスの値にする

            score = self.metrics(va_y, va_pred)

            # =================================
            """
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
            """
            # =================================

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, va_pred_prob, score
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
        preds_prob = []  # 各foldの推論結果を保存

        # 各foldで学習を行う
        for i_fold in range(self.n_splits):
            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, va_pred_prob, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model(self.out_dir_name)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)
            preds_prob.append(va_pred_prob)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]
        preds_prob = np.concatenate(preds_prob, axis=0)
        preds_prob = preds_prob[order]

        # 全体のスコアを算出（これを平均とする）
        all_mean_score = self.metrics(self.train_y, preds)

        # =================================
        # csvに書き込む
        self.score_list.append(['mean_score', all_mean_score])
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

        self.logger.info(f'{self.run_name} - end training cv - mean score {all_mean_score}')

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(preds), self.out_dir_name + f'{self.run_name}_train.pkl')
            Util.dump_df_pickle(pd.DataFrame(preds_prob), self.out_dir_name + f'{self.run_name}_train_prob.pkl')

        # 評価結果の保存
        self.logger.result_scores(self.run_name, scores, all_mean_score)

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance()

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        test_x = self.load_x_test()
        preds = []
        preds_prob = []  # 各foldの推論結果を保存

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model(self.out_dir_name)
            pred = model.predict(test_x)
            pred_prob = pred
            pred = np.argmax(pred, axis=1)  # 最尤と判断したクラスの値にする
            preds.append(pred)
            preds_prob.append(pred_prob)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)
        pred_prob_avg = np.mean(preds_prob, axis=0)

        # 推論結果の保存（submit対象データ）
        pred_prob_avg_df = pd.DataFrame(pred_prob_avg)
        pred_prob_avg_df = pred_prob_avg_df.rename(columns={0: '79ad4647da6de6425abf_850',
                                                            1: '79ad4647da6de6425abf_780',
                                                            2: 'c695a1e61e002b34e556_780',
                                                            3: '6718e7f83c824b1e436d_780',
                                                            4: '0b9dbf13f938efd5717f_780',
                                                            5: '84b788fdc5e779f8a0df_850',
                                                            6: '118c70535bd753a86615_780',
                                                            7: '118c70535bd753a86615_850',
                                                            8: '0b9dbf13f938efd5717f_850',
                                                            9: '6718e7f83c824b1e436d_850',
                                                            })
        pred_prob_avg_df = pred_prob_avg_df.loc[:, ['0b9dbf13f938efd5717f_780', '0b9dbf13f938efd5717f_850',
                                                    '118c70535bd753a86615_780', '118c70535bd753a86615_850',
                                                    '6718e7f83c824b1e436d_780', '6718e7f83c824b1e436d_850',
                                                    '79ad4647da6de6425abf_780', '79ad4647da6de6425abf_850',
                                                    '84b788fdc5e779f8a0df_850', 'c695a1e61e002b34e556_780',
                                                    ]]
        Util.dump_df_pickle(pd.DataFrame(pred_avg), self.out_dir_name + f'{self.run_name}_pred.pkl')
        Util.dump_df_pickle(pred_prob_avg_df, self.out_dir_name + f'{self.run_name}_pred_prob.pkl')

        self.logger.info(f'{self.run_name} - end prediction cv')