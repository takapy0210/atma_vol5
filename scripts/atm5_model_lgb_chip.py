import lightgbm as lgb
from model_lgb import ModelLGB

# 各foldのモデルを保存する配列
model_array = []


class atm5_ModelLGB(ModelLGB):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        dtrain = lgb.Dataset(tr_x, tr_y, categorical_feature=self.categoricals)

        if validation:
            dvalid = lgb.Dataset(va_x, va_y, categorical_feature=self.categoricals)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        verbose_eval = params.pop('verbose_eval')

        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = lgb.train(
                                params,
                                dtrain,
                                num_boost_round=num_round,
                                valid_sets=(dtrain, dvalid),
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=verbose_eval,
                                )
            model_array.append(self.model)

        else:
            watchlist = [(dtrain, 'train')]
            self.model = lgb.train(params, dtrain, num_boost_round=num_round, evals=watchlist)
            model_array.append(self.model)
