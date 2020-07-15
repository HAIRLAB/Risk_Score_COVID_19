# -- coding:utf-8 --
import numpy as np
import pandas as pd
from scipy.special import expit, logit


def score_form(x: np.array):
    """
    特征顺序：乳酸脱氢酶,  超敏C反应蛋白,  淋巴细胞(%)
    w = [0.0025772413536092565, 0.0064209452664954324, -0.03172279948348779, -0.8438356708513047]
    :param x:
    :return:

    唐秀川 2020.04.07
    """
    x = x.copy()

    # 模型参数
    mean = [360.19693396, 58.82753538, 19.22334906]
    std = [307.76450909, 86.4720017, 13.44904507]
    coef = [2.61753186, 1.76451706, -2.01219637]
    intercept = -0.91332987

    # 计算特征等效权重，decision_function(X) = x1 * w1 + x2 * w2 + x3 * w3 + b
    w = [coef[i]/std[i] for i in range(3)] + [intercept - np.sum([coef[i] * mean[i] / std[i] for i in range(3)])]

    # 使决策函数结果变化 0.2 的步长
    decision_function_step = 0.2
    rs_step  =  decision_function_step / w[0]
    cmc_step =  decision_function_step / w[1]
    lb_step  = -decision_function_step / w[2]

    rs_range  = mean[0] + rs_step  * np.arange(-10, 13)
    cmc_range = mean[1] + cmc_step * np.arange(-5, 12)
    lb_range  = mean[2] + lb_step  * np.arange(-11, 12)
    # print(f"乳酸脱氢酶：{rs_range.round(0)}")
    # print(f"超敏C反应蛋白：{cmc_range.round(2)}")
    # print(f"淋巴细胞：{lb_range.round(2)}")

    # 分数 -> 死亡概率
    decision_function0 = np.sum([mean[i] * w[i] for i in range(3)]) + w[3]   # 0 分对应的决策函数值
    prob0 = expit(decision_function0)                                        # probability: sigmoid(decision_function0)
    prob_df = pd.DataFrame(
        [[i, decision_function0 + i * decision_function_step, expit(decision_function0 + i * decision_function_step).round(2)]
         for i in range(-150, 260)],
        columns=['score', 'decision_function', 'probability']
    )
    # print(prob_df)

    x[:, 0] = pd.cut(
        x[:, 0],
        [-2] + list(rs_range) + [1e5],
        labels=list(range(-10, 14))           # a, b+1
    )
    x[:, 1] = pd.cut(
        x[:, 1],
        [-2] + list(cmc_range) + [1e5],
        labels=list(range(-5, 13))           # a, b+1
    )
    x[:, 2] = pd.cut(
        x[:, 2],
        [-2] + list(lb_range) + [1e5],
        labels=list(range(12, -12, -1))      # -(a-1), b
    )
    x = x.sum(axis=1)

    pred = (x > 4).astype(int)
    score_prob_dict = prob_df.set_index('score')['probability'].to_dict()
    prob = pd.Series(x).map(score_prob_dict).values
    return x, pred, prob


if __name__ == '__main__':
    score_form(np.array([[550, 50, 5.1],
                         [450, 40, 4.1]]))