# -- coding:utf-8 --
import numpy as np
import pandas as pd

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import auc, roc_curve
import utils
import warnings

from generate_score_form import score_form

warnings.filterwarnings("ignore")


class LR_model:
    def __init__(self):
        # 根据孙川训练的模型，手动定义
        self.mean = [360.19693396, 58.82753538, 19.22334906]
        self.std = [307.76450909,  86.4720017,  13.44904507]
        self.coef = [2.61753186, 1.76451706, -2.01219637]
        self.intercept = -0.91332987

    def _predict_proba_lr(self, x: np.array):
        # 归一化后，计算 proba，输出的二分类的概率
        pred_score = (x[:, 0]-self.mean[0])/self.std[0]*self.coef[0] + \
                     (x[:, 1]-self.mean[1])/self.std[1]*self.coef[1] + \
                     (x[:, 2]-self.mean[2])/self.std[2]*self.coef[2] + self.intercept
        pred_score = 1. / (1. + np.exp(-pred_score))

        return np.concatenate([(1-pred_score).reshape(-1,1), pred_score.reshape(-1,1)], axis=1)

    def predict(self, x: np.array):
        # 归一化后，计算 proba，并输出标签
        pred_score = (x[:, 0] - self.mean[0]) / self.std[0] * self.coef[0] + \
                     (x[:, 1] - self.mean[1]) / self.std[1] * self.coef[1] + \
                     (x[:, 2] - self.mean[2]) / self.std[2] * self.coef[2] + self.intercept

        pred = pd.Series(1. / (1. + np.exp(-pred_score)))
        pred = pred.apply(lambda x: 0 if x < 0.5 else 1)

        return pred.values


def preprocess_tongji_data(features):
    """
    预处理同济医院数据集

    郭裕祺  2020.04.26
    """
    data = pd.read_parquet('./data/time_series_1559.parquet')

    # 滑窗合并，去除三特征有缺失的项
    data = utils.merge_data_by_sliding_window(data, n_days=1, dropna=False, time_form='diff')
    data = data.sort_index(level=(0, 1), ascending=False)
    data = data.reset_index()
    data = data.dropna(how='any', subset=features)

    return data


def show_confusion_matrix(validations, predictions, path='confusion.png'):
    """
    绘制混淆矩阵

    郭裕祺  2020.04.02
    """
    LABELS = ['Survival', 'Death']
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(path, dpi=500, bbox_inches='tight')
    plt.show()


def compute_auc_all(data, model, features, days=10):
    """
    计算每个时间段的auc

    郭裕祺  2020.04.26
    """
    day_list = list(range(0, days+1))
    sample_num = []
    survival_num = []
    death_num = []
    auc_ = []
    precision = []
    recall = []
    add_before_auc = []
    for i in range(0, days+1):

        if i == 0:

            # 有的病人出院或死亡后的几个小时最后一次检测才会出来
            # data_subset 第 i 天的数据
            data_subset     = data.loc[data['t_diff'] <= 0].groupby('PATIENT_ID').last()
            data_subset_sum = data.loc[data['t_diff'] <= 0]

        else:

            # data_subset_sum 是 <= i 的数据
            data_subset     = data.loc[data['t_diff'] == i].groupby('PATIENT_ID').last()
            data_subset_sum = data.loc[data['t_diff'] <= i]

        # 统计对应子集的结果
        if data_subset.shape[0] > 0:

            sample_num  .append(data_subset.shape[0])
            survival_num.append(sum(data_subset['出院方式'] == 0))
            death_num   .append(sum(data_subset['出院方式'] == 1))

            # 计算 auc
            fpr, tpr, threshold = roc_curve(
                data_subset['出院方式'].values,
                model._predict_proba_lr(data_subset[features].values)[:,1]
            )
            auc_.append(auc(fpr, tpr))

            # 计算查准率和查全率
            precision.append(precision_score(data_subset['出院方式'].values, model.predict(data_subset[features].values)))
            recall.append   (recall_score   (data_subset['出院方式'].values, model.predict(data_subset[features].values)))

            # 计算 auc
            fpr, tpr, threshold = roc_curve(
                data_subset_sum['出院方式'].values,
                model._predict_proba_lr(data_subset_sum[features].values)[:, 1]
            )
            add_before_auc.append(auc(fpr, tpr))

        else:

            sample_num.append(np.nan)
            survival_num.append(np.nan)
            death_num.append(np.nan)
            auc_.append(np.nan)
            precision.append(np.nan)
            recall.append(np.nan)
            add_before_auc.append(np.nan)

    return day_list, auc_, precision, recall, sample_num, survival_num, death_num, add_before_auc


def plot_auc_time(test_data, features, model, days=18, save_path=None):
    """
    绘制模型在不同时间段数据上的auc

    郭裕祺  2020.04.26
    """
    test_model_result = pd.DataFrame()
    test_model_result['day'], test_model_result['auc_score'], test_model_result['precision-score'], \
    test_model_result['recall-score'], test_model_result['sample_num'], test_model_result['survival_num'], \
    test_model_result['death_num'], test_model_result['add_before_auc'] = compute_auc_all(test_data, model, features, days=days)

    # 画auc-时间曲线
    fig = plt.figure(figsize=(8, 6))
    plt.tick_params(labelsize=20)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    n1 = ax.bar  (test_model_result['day'], test_model_result['sample_num'],   label='Death',    color='red',        alpha=0.5, zorder=0)
    n2 = ax.bar  (test_model_result['day'], test_model_result['survival_num'], label='Survival', color='lightgreen', alpha=1,   zorder=5)
    p1 = ax2.plot(test_model_result['day'], test_model_result['auc_score'],      marker='o', linestyle='-', color='black', label='auc',            zorder=10)
    p2 = ax2.plot(test_model_result['day'], test_model_result['add_before_auc'], marker='o', linestyle='-', color='blue',  label='cumulative auc', zorder=10)

    fig.legend(loc='center left', bbox_to_anchor=(0.2, 1.2), bbox_transform=ax.transAxes, fontsize=20)

    ax.set_xlabel('days to outcome', fontsize=20)
    ax2.set_ylabel('auc', fontsize=20)
    ax.set_ylabel('sample_num', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xticks(list(range(0, days+1, 2)))
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()


def get_survival_rate(data_target, days=30):
    """
    计算不同时间点的生存率，用于画km图

    郭裕祺  2020.04.26
    """
    sample_num = data_target.shape[0]
    death_num = [0]
    for day in range(1, days+1):
        death_num.append(sum((data_target['t_diff'] <= day) & (data_target['出院方式'] == 1)))
    survival_rate = pd.DataFrame()
    survival_rate['day']           = list(range(0, days+1))
    survival_rate['survival_rate'] = death_num
    survival_rate['survival_rate'] = (sample_num - survival_rate['survival_rate']) / sample_num
    return survival_rate, sample_num


def plot_Kaplan_Meier(data_group, str_group, days=30, save_path='KM.png'):
    """
    绘制给定的三组样本的生存率曲线（km图）

    郭裕祺  2020.04.26
    """
    # 计算三组样本在不同时间点的生存率，以及各组的样本数
    survival_rate_mild,   mild_num   = get_survival_rate(data_group[0], days)
    survival_rate_med,    med_num    = get_survival_rate(data_group[1], days)
    survival_rate_severe, severe_num = get_survival_rate(data_group[2], days)
    total_num = mild_num + med_num + severe_num

    # 打印最终的死亡率
    print(f"{save_path}\n"
          f"mild final death rate: {100 - survival_rate_mild['survival_rate'].iat[-1]*100:.1f}%\n"
          f"med final death rate: {100 - survival_rate_med['survival_rate'].iat[-1]*100:.1f}%\n"
          f"severe final death rate: {100 - survival_rate_severe['survival_rate'].iat[-1]*100:.1f}%")
    print(f'{mild_num / total_num * 100:.1f} percent of the patients were in the low-risk group, '
          f'{med_num / total_num * 100:.1f} percent of the patients were in the intermediate-risk group, '
          f'{severe_num / total_num * 100:.1f} percent of the patients were in the high-risk group.')
    print(f"患者总数：{total_num}")

    # 画KM曲线图
    mild_label   = "low-risk"   + str_group[0] + ": {0} patients".format(mild_num)
    med_label    = "intermediate-risk" + str_group[1] + ": {0} patients".format(med_num)
    severe_label = "high-risk" + str_group[2] + ": {0} patients".format(severe_num)

    plt.step(survival_rate_mild  ['day'], survival_rate_mild  ['survival_rate'], c='lightgreen', label=mild_label)
    plt.step(survival_rate_med   ['day'], survival_rate_med   ['survival_rate'], c='blue',       label=med_label)
    plt.step(survival_rate_severe['day'], survival_rate_severe['survival_rate'], c='red',        label=severe_label)
    plt.legend(loc='center left', bbox_to_anchor=(0.02, 1.15), fontsize=14)
    plt.xlabel('days', fontsize=14)
    plt.ylabel('Survival rate', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim((0, 1.05))
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()


def plot_lr_km(data, model, save_path='image/KM.png'):
    """
    绘制SVC模型在给定测试数据集上的KM曲线

    郭裕祺  2020.04.26
    """
    # 取每个人最早的数据
    data_first = data.groupby('PATIENT_ID').first()

    # 模型给出预测分数和预测类别
    data_first['pred_score'] = model._predict_proba_lr(data_first[features].values)[:, 1]
    data_first['pred']       = model.predict(data_first[features].values)

    # 按照预测分数划分轻、中、危三组。这里划分的根据是 'pred_score' 的区域，而不是医生的定义
    thresh = [0.3, 0.98]
    mild_data   = data_first.loc[data_first['pred_score'] < thresh[0]]
    med_data    = data_first.loc[(data_first['pred_score'] >= thresh[0]) & (data_first['pred_score'] < thresh[1])]      # 0.3， 0.5
    severe_data = data_first.loc[data_first['pred_score'] >= thresh[1]]                                           # 0.5
    data_group  = [mild_data, med_data, severe_data]
    str_group   = [f'(score<{thresh[0]})', f'({thresh[0]}<=score<{thresh[1]})', f'(score>={thresh[1]})']

    # 画KM曲线
    plot_Kaplan_Meier(data_group, str_group, days=30, save_path=save_path)


def CRB_65(x: pd.Series):
    """CRB-65 打分
    example: df.apply(CRB_65, axis=1)

    :param x: 单个样本，['年龄', '呼吸', 'SBP', 'DBP', '神智']
    :return: score 0 低危；1~2 中危；>=3 高危
    """
    score = 0
    if x['年龄'] >= 65:
        score += 1

    if x['呼吸'] >= 30:
        score += 1

    if (x['SBP'] < 90) or (x['DBP'] <= 60):
        score += 1

    if x['神智'] > 1:
        score += 1

    return score


def CURB_65(x: pd.Series):
    """CURB-65 打分
    example: df.apply(CRB_65, axis=1)

    :param x: 单个样本，['年龄', '呼吸', 'SBP', 'DBP', '神智', '尿素']
    :return: score 0 低危；1~2 中危；>=3 高危
    """
    score = 0
    if x['年龄'] >= 65:
        score += 1

    if x['呼吸'] >= 30:
        score += 1

    if (x['SBP'] < 90) or (x['DBP'] <= 60):
        score += 1

    if x['神智'] > 1:
        score += 1

    if x['尿素'] > 7:
        score += 1

    return score


def qSOFA(x: pd.Series):
    """CRB-65 打分
    example: df.apply(CRB_65, axis=1)

    :param x: 单个样本，['年龄', '呼吸', 'SBP', '神智']
    :return: score 0~4
    """
    score = 0
    if x['年龄'] >= 65:
        score += 1

    if x['呼吸'] >= 22:
        score += 1

    if x['SBP'] <= 100:
        score += 1

    if x['神智'] > 1:
        score += 1

    return score


def preprocess_comparison_data():
    """
    处理得到用于比较不同打分方法的数据集

    郭裕祺  2020.04.26
    """
    features = [
        '乳酸脱氢酶',
        '超敏C反应蛋白',
        '淋巴细胞(%)',
        '尿素',
        '年龄',
        '神智',
        'SBP',
        'DBP',
        '呼吸',
    ]

    data = pd.read_parquet('./data/time_series_1559.parquet')

    # 滑窗合并，去除特征有缺失的项，取每个人最早的数据
    data = utils.merge_data_by_sliding_window(data, n_days=1, dropna=False, time_form='diff')
    data = data.sort_index(level=(0, 1), ascending=False)
    data = data.reset_index()
    data = data.dropna(how='any', subset=features)

    return data.groupby('PATIENT_ID').first()


def roc_comparison(model):
    """
    对比SVC模型和其他传统打分算法的roc曲线

    郭裕祺  2020.04.26
    """
    # 获取数据集
    data_comparison = preprocess_comparison_data()

    # 计算SVC打分
    data_comparison['pred_score'] = model._predict_proba_lr(data_comparison[features].values)[:, 1]
    data_comparison['pred']       = model.predict          (data_comparison[features].values)

    # 计算传统打分
    data_comparison['CRB_65']  = data_comparison[['年龄', '神智', 'SBP', 'DBP', '呼吸']].apply(CRB_65, axis=1)
    data_comparison['CURB_65'] = data_comparison[['年龄', '神智', 'SBP', 'DBP', '呼吸', '尿素']].apply(CURB_65, axis=1)
    data_comparison['qSOFA']   = data_comparison[['年龄', '神智', 'SBP', '呼吸']].apply(qSOFA, axis=1)

    # 绘制ROC曲线对比图
    # SVC
    fpr, tpr, threshold = roc_curve(data_comparison['出院方式'].values, data_comparison['pred_score'].astype(float).values)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='LR auc: %.4f' % auc_, c='darkred')

    # CRB_65
    fpr, tpr, threshold = roc_curve(data_comparison['出院方式'].values, data_comparison['CRB_65'].astype(float).values)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='CRB_65 auc: %.4f' %auc_, c='orange')

    # CURB_65
    fpr, tpr, threshold = roc_curve(data_comparison['出院方式'].values, data_comparison['CURB_65'].astype(float).values)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='CURB_65 auc: %.4f' %auc_, c='deepskyblue')

    # qSOFA
    fpr, tpr, threshold = roc_curve(data_comparison['出院方式'].values, data_comparison['qSOFA'].astype(float).values)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='qSOFA auc: %.4f' %auc_, c='green')

    plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.2), fontsize=14)

    # 对角线
    plt.plot([0, 1], [0, 1], c='grey', linestyle=':')
    plt.xlabel('1-Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('image/同济/roc.png', dpi=500, bbox_inches='tight')
    plt.show()


def score_pred_hist(data, model, p_save_part, bins=(20, 20)):
    # 准备数据
    data = data.groupby('PATIENT_ID').first()
    score, pred, prob = model(data[features].values)
    data['score'] = score
    data['prob'] = prob

    fs = 12         # 图片中文字大小
    survival = data.query("`出院方式` == 0")
    death    = data.query("`出院方式` == 1")

    # 打分分布图
    survival['score'].plot.hist(bins=bins[0], alpha=0.5, label='Survival', fontsize=fs)
    death['score'].plot.hist(bins=bins[1], alpha=0.5, label='Death', fontsize=fs)
    plt.xlabel('Score', fontsize=fs)
    plt.ylabel('Count', fontsize=fs)
    plt.title('Score distribution', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig(f"{p_save_part}_score_dist.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # 对应概率分布图
    survival['prob'].plot.hist(bins=bins[0], alpha=0.5, label='Survival', fontsize=fs)
    death['prob'].plot.hist(bins=bins[1], alpha=0.5, label='Death', fontsize=fs)
    plt.xlabel('Probability', fontsize=fs)
    plt.ylabel('Count', fontsize=fs)
    plt.title('Probability distribution', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig(f"{p_save_part}_prob_dist.png", bbox_inches='tight', dpi=300)
    plt.close('all')


def jyt_test(model, features):
    """
    绘制SVC模型在金银潭数据上的cumulative AUC图，KM曲线, roc曲线、混淆矩阵（以0.5为分界点）

    郭裕祺  2020.04.26
    """
    jyt_data = pd.read_parquet('./data/金银潭100/jyt100.parquet').reset_index()
    jyt_data2 = pd.read_parquet('./data/金银潭100/jyt46.parquet').reset_index().dropna(subset=['出院时间'])
    jyt_data['t_diff'] = (jyt_data['出院时间'] - jyt_data['RE_DATE']).dt.days
    jyt_data2['t_diff'] = (jyt_data2['出院时间'].dt.normalize() - jyt_data2['RE_DATE'].dt.normalize()).dt.days
    jyt_data = jyt_data.append(jyt_data2)

    # 画 AUC 随时间变化曲线，cumulative AUC
    plot_auc_time(jyt_data, features, model, days=21, save_path='./image/金银潭/jyt_auc_time.png')

    # 画 KM 曲线，取每个人第一次数据
    plot_lr_km(jyt_data, model, save_path='./image/金银潭/jyt_KM.png')

    # 画roc曲线，取第一次数据
    jyt_data               = pd.read_parquet('./data/金银潭100/jyt100.parquet')
    jyt_data2              = pd.read_parquet('./data/金银潭100/jyt46.parquet')
    jyt_data               = jyt_data.append(jyt_data2).sort_index()
    jyt_data               = jyt_data.groupby('PATIENT_ID').first()
    jyt_data['pred']       = model.predict          (jyt_data[features].values)
    jyt_data['pred_score'] = model._predict_proba_lr(jyt_data[features].values)[:, 1]

    fpr, tpr, threshold = roc_curve(jyt_data['出院方式'].values, jyt_data['pred_score'].astype(float).values)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='jyt auc: %.4f' % auc_, c='red')
    plt.legend(fontsize=14)
    plt.plot([0, 1], [0, 1], c='grey', linestyle=':')
    plt.title('Roc curve of Jinyintan hospital', fontsize=14)
    plt.xlabel('1-Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('image/金银潭/jyt_roc.png', dpi=500, bbox_inches='tight')
    plt.show()

    # 分数和预测概率的直方图
    score_pred_hist(jyt_data, score_form, './image/金银潭/jyt', bins=(15, 15))

    # 画混淆矩阵
    print('jyt_f1: {}'.format(f1_score(jyt_data['出院方式'].values, jyt_data['pred'].values)), '\n\n')
    show_confusion_matrix(jyt_data['出院方式'].values, jyt_data['pred'].values, path='image/金银潭/jyt_confusion.png')


def sz_test(model, features):
    """
    绘制SVC模型在深圳数据上的cumulative AUC图，KM曲线, roc曲线、混淆矩阵（以0.5为分界点）

    孙川  2020.05.04
    """
    # 画 AUC 随时间变化曲线，cumulative AUC
    sz_data = pd.read_parquet('./data/深圳/sz.parquet').reset_index().dropna(subset=['出院时间'])
    sz_data['t_diff'] = (sz_data['出院时间'].dt.normalize() - sz_data['RE_DATE'].dt.normalize()).dt.days
    plot_auc_time(sz_data, features, model, days=21, save_path='./image/深圳/sz_auc_time.png')

    # 画 KM 曲线，取每个人第一次数据
    plot_lr_km(sz_data, model, save_path='./image/深圳/sz_KM.png')

    # 画roc曲线，取第一次数据
    sz_data               = pd.read_parquet('./data/深圳/sz.parquet')
    sz_data               = sz_data.groupby('PATIENT_ID').first()
    sz_data['pred']       = model.predict          (sz_data[features].values)
    sz_data['pred_score'] = model._predict_proba_lr(sz_data[features].values)[:, 1]

    fpr, tpr, threshold = roc_curve(sz_data['出院方式'].values, sz_data['pred_score'].astype(float).values)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='sz auc: %.4f' % auc_, c='red')
    plt.legend(fontsize=14)
    plt.plot([0, 1], [0, 1], c='grey', linestyle=':')
    plt.title('Roc curve of Shenzhen hospital', fontsize=14)
    plt.xlabel('1-Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('image/深圳/sz_roc.png', dpi=500, bbox_inches='tight')
    plt.show()

    # 分数和预测概率的直方图
    score_pred_hist(sz_data, score_form, './image/深圳/sz', bins=(20, 2))

    # 画混淆矩阵
    print('sz_f1: {}'.format(f1_score(sz_data['出院方式'].values, sz_data['pred'].values)))
    show_confusion_matrix(sz_data['出院方式'].values, sz_data['pred'].values, path='image/深圳/sz_confusion.png')


if __name__ == '__main__':
    features = [
        '乳酸脱氢酶',
        '超敏C反应蛋白',
        '淋巴细胞(%)',
    ]
    add_features = ['尿素']

    # 获取同济医院生化指标数据集
    data = preprocess_tongji_data(features)

    # 导入模型
    model = LR_model()

    # 在1479人训练集上画auc随时间变化曲线
    plot_auc_time(data, features, model, days=21, save_path='image/同济/tj_auc_time.png')

    # 在1479人训练集绘制KM曲线 取每个人第一次数据
    plot_lr_km(data, model, save_path='./image/同济/tj_KM.png')

    # 对比SVC模型和其他传统打分算法的roc曲线
    roc_comparison(model)

    # 分数和预测概率的直方图
    score_pred_hist(data, score_form, './image/同济/tj')

    # 测试模型在其他中心的数据上的表现
    # 金银潭
    jyt_test(model, features)
    # 深圳
    sz_test(model, features)



