# 类/函数 与论文的对应关系
## EDA
- Model: 同济数据集训练，金银潭数据集验证，输出 StandardScaler 和 LR 参数

## generate_score_form
- score_form: 生成打分表，函数中的 w 变量为等效权重，对应论文中的死亡概率公式；打印出的结果对应论文最后打分表部分

## svc_score
- LR_model: 实现 StandardScaler 和 LR，效果与 EDA 中的相同
- show_confusion_matrix: 画混淆矩阵
- compute_auc_all: 计算每个时间段的 AUC
- plot_auc_time: 绘制不同时间段的 AUC，对应着论文中图1、S3、S6
- get_survival_rate: 计算不同时间点的生存率，用于画km图
- plot_Kaplan_Meier: 绘制K-M图
- plot_svc_km: 绘制 LR 模型在给定测试数据集上的KM曲线，对应着论文中图2、S2、S5；还会输出一些比例，这些数据散落在文章中。
- CRB_65、CURB_65、qSOFA: 计算相关医学指标
- preprocess_tongji_data: 预处理统计数据
- preprocess_comparison_data: 预处理得到用于比较不同打分方法的数据集
- roc_comparison: 对比 LR 模型和其他传统打分算法的 ROC 曲线，对应论文中图3，图中的 AUC 数值散落在论文中
- jyt_test: 针对金银潭测试集，调用上述相关功能函数，产生图片和数据
- sz_test: 针对深圳测试集，调用上述相关功能函数，产生图片和数据

## utils
通用函数