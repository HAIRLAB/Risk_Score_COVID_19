# -- coding:utf-8 --
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import utils


class Model:
    """求解模型参数
    用 1559 数据做训练集，标准化后训练 LinearSVC 模型，得到标准化参数和模型参数

    孙川  2020.04.26
    """
    def __init__(self):
        self.seed = 0
        self.model = LogisticRegression(random_state=self.seed, solver='liblinear')

    def _load_data(self, in_out_time, return_raw_test=False):
        # 读取训练集数据
        xy_tr = utils.read('./data/time_series_1559.parquet', usecols=utils.top3_feats_cols + utils.in_out_time_cols + ['出院方式'])

        # 对训练集进行滑窗合并
        xy_tr = utils.merge_data_by_sliding_window(xy_tr, n_days=1, dropna=True, subset=utils.top3_feats_cols, time_form='diff')

        # 取训练集中，距离出院天数小于 10 的检测结果
        xy_tr = xy_tr[(xy_tr.reset_index(level=1)['t_diff'] < 10).values].droplevel(1)
        print(f"训练集数据量：{len(xy_tr)}")

        # 将训练集划分为模型输入和标签
        x_tr = xy_tr[utils.top3_feats_cols]
        y_tr = xy_tr['出院方式']

        # 读取金银潭测试集数据
        xy_te_raw = pd.read_parquet('./data/金银潭100/jyt100.parquet').sort_index(level=[0, 1], ascending=False)

        # 提取测试集最接近入院数据
        xy_te = xy_te_raw.groupby('PATIENT_ID').last()  # 取最早的检测结果，diff 越大越早

        # 读取二医院测试集数据
        # xy_te_raw = pd.read_parquet('./data/二医院/2yy502.parquet').dropna(subset=utils.top3_feats_cols)
        # xy_te = xy_te_raw

        # 将测试集划分为模型输入和标签
        x_te = xy_te[utils.top3_feats_cols]
        y_te = xy_te['出院方式']

        if return_raw_test:
            return x_tr, x_te, y_tr, y_te, xy_te_raw
        else:
            return x_tr, x_te, y_tr, y_te

    def devel(self):
        # 加载数据
        x_tr, x_te, y_tr, y_te = self._load_data('n_days')

        # 标准化
        ss = StandardScaler()
        x_tr = ss.fit_transform(x_tr)
        x_te = ss.transform(x_te)
        print(f"数据标准化：均值： {ss.mean_}, 标准差：{ss.scale_}")

        # 训练模型并预测
        clf = self.model
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_te)
        print(f"LogisticRegression 模型系数：{clf.coef_}，截距：{clf.intercept_}")

        # 输出相关指标
        metrics = utils.Metrics(report='overall', acc='overall', conf_mat='overall')
        metrics.record(y_te, pred)
        metrics.print_metrics()


if __name__ == '__main__':
    Model().devel()
