# 本次初步复现结果

完成日期：2026-10-21（Asia/Shanghai）。数据：MARC；任务：English→Chinese五星评分分类。

## 可直接向老师展示的结果

| 方法 | 英文测试Accuracy | 中文测试Accuracy | 中文Macro-F1 | 中文MAE | 中文ECE校准前 | 中文ECE校准后 |
|---|---:|---:|---:|---:|---:|---:|
| mBERT | 59.50% | 35.64% | 0.3299 | 0.9536 | 0.1008 | 0.0539 |
| XLM-R Base | 61.88% | 49.90% | 0.4951 | 0.6574 | 0.1338 | 0.0870 |
| XLM-R + DANN | 61.84% | 52.28% | 0.5211 | 0.6128 | 0.1041 | 0.0592 |

MAE、ECE越低越好。温度缩放不改变预测类别，因此表中Accuracy同时适用于校准前后。

本次观察：XLM-R相对mBERT的中文准确率增加14.26个百分点。DANN相对XLM-R增加2.38个百分点，中文MAE减少0.0446。三个模型的中文ECE在英文校准后均降低。这些都是一次固定种子pilot的观察，不能作为稳定提升或统计显著性的结论。

温度参数：mBERT 1.268669、XLM-R 1.231322、XLM-R+DANN 1.221766。仅使用英文校准集拟合。

## 四篇分别看哪里

| 论文 | 文件 | 本次范围 |
|---|---|---|
| 1 MARC | `reports/marc_en_zh_pilot/paper1_results.csv` | mBERT的英文监督测试与中文zero-shot测试；训练预算缩小 |
| 2 XLM-R | `reports/marc_en_zh_pilot/paper2_results.csv` | 公开XLM-R权重在MARC的下游迁移；MARC并非原XLM-R论文实验 |
| 3 DANN | `reports/marc_en_zh_pilot/paper3_results.csv` | XLM-R source-only与DANN比较；语言作域、五星任务，属于方法扩展 |
| 6 Calibration | `reports/marc_en_zh_pilot/paper6_results.csv` | 三个模型中英文测试的校准前后对比 |

更适合报告的表述：**统一中英MARC设置下的基准pilot、XLM-R下游实验、DANN方法扩展与Temperature Scaling方法复现。**

## 本次配置与数据权限

- 训练：同一批分层抽样的10,000条英文，五个星级各2,000条。
- DANN额外使用10,000条随机抽样的中文训练文本，输入文件中不含中文标签。
- 原英文validation固定拆分为模型选择2,500条、校准2,500条。
- 英文/中文test各5,000条，仅最终评估；中文validation未使用。
- 每模型2轮、seed42、最大长度128、FP16、RTX 4060 Laptop 8GB。
- 完整原始中英文数据共420,000条仍保存在`data/raw/marc/`，后续可以扩大训练规模。

## 资源记录

| 模型 | 实际参数量 | 训练分钟 | PyTorch峰值allocated显存GB |
|---|---:|---:|---:|
| mBERT | 177,266,693 | 3.25 | 3.60 |
| XLM-R | 277,456,901 | 4.10 | 6.01 |
| XLM-R+DANN | 277,654,279 | 5.45 | 6.03 |

三个训练阶段合计约12.8分钟，不包含首次下载、安装、模型加载、分词、最终评估和出图。显存列为PyTorch分配峰值，系统显示显存还包括CUDA上下文、缓存和其他应用。

## 证据与复核

- `reports/marc_en_zh_pilot/verification.json`：128项检查通过，包括数据划分、文件哈希、预测数量以及从logits独立重算所有指标。
- `tests/test_research_logic.py`：4项测试通过，覆盖梯度反转、已知指标、空标题处理和温度优化。
- `results/marc_en_zh_pilot/<model>/`：逐条预测CSV、测试及校准logits、指标JSON、温度和逐轮训练记录。
- `models/finetuned/marc_en_zh_pilot/<model>/best.safetensors`：各模型最佳权重，按英文模型选择集MAE选择。
- `environment/requirements-lock.txt`、`environment/hardware.json`：实际依赖和硬件。
- `data/provenance/downloads.json`：所有公开资产的固定版本、URL、文件字节数和SHA256。
- `logs/last_run_status.json`：最终运行退出码0。

原始数据中的少量空标题统一转为空字符串，未删减样本；null计数保留在数据audit中。原始文件保持完整。

## 图表阅读提示

Reliability diagram中某些置信度区间样本极少，曲线可能有较大局部波动；各区间的count在metrics.json的reliability_bins中。ECE按区间样本数加权。低ECE不代表高分类准确率，应同时看Accuracy、MAE和NLL。

这一步只检验中英文迁移，没有单独控制产品类别。中文训练集中book占31.529%，英文为1.878%，因此迁移同时伴随产品类别分布变化。下一轮研究需要单独设计语言/类别控制实验及多随机种子，而不应仅凭本表声称已解决跨域泛化。
