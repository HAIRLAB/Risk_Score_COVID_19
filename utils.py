# -- coding:utf-8 --
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 常用参数
top3_feats_cols = ['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)']
in_out_time_cols = ['入院时间', '出院时间']


def is_number(s):
    if s is None:
        s = np.nan

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=None, time_form='diff'):
    """滑窗合并数据

    :param data: 时间序列数据，一级行索引为 PATIENT_ID, 二级行索引为 RE_DATE
    :param n_days: 窗口长度
    :param dropna: 滑窗合并后还缺失的是否删掉
    :param subset: pd.DataFrame().dropna() 参数                                                   Note: 新参数!
    :param time_form: 返回数据的时间索引，'diff' or 'timestamp'
    :return: 合并后的数据，一级行索引为 PATIENT_ID, 二级行索引为 t_diff or RE_DATE, 取决于"time_form"
    """
    data = data.reset_index(level=1)

    # 距离出院时长        Note: 去掉了出院时间和检测时间的时分秒，因为我觉得以 00:00:00 为分界点更合适
    t_diff = data['出院时间'].dt.normalize() - data['RE_DATE'].dt.normalize()
    data['t_diff'] = t_diff.dt.days.values // n_days * n_days
    data = data.set_index('t_diff', append=True)

    # 滑窗合并
    data = (
        data
        .groupby(['PATIENT_ID', 't_diff']).ffill()
        .groupby(['PATIENT_ID', 't_diff']).last()
    )

    if dropna:
        data = data.dropna(subset=subset)         # Note: 这里对缺失值进行了 dropna(), 而不是 fillna(-1)

    if time_form == 'timestamp':
        data = (
            data
            .reset_index(level=1, drop=True)
            .set_index('RE_DATE', append=True)
        )
    elif time_form == 'diff':
        data = data.drop(columns=['RE_DATE'])

    return data
