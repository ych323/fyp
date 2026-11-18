# 初步复现实验协议 / Pilot protocol

## 本次运行

- MARC英文→中文，五星分类，标签内部编码0–4，导出时为1–5星。
- 一次随机种子42。每个模型同一份10,000条英文分层样本，各星级2,000条。
- mBERT、XLM-R、XLM-R+DANN各训练2轮，总长度128，英文body最多110tokens，然后配对title+category；超过总长时longest-first截断。
- 学习率2e-5，AdamW，weight decay 0.01，warmup 10%，梯度裁剪1.0。
- 源语言microbatch=8，梯度累计4，源语言有效batch=32。DANN每个microbatch另加8条无标签中文；任务损失仅使用英文。
- 模型选择集为原英文validation中的2,500条；按MAE最小选择checkpoint，不使用中文标签选择模型或超参数。
- 剩余2,500条英文validation只拟合temperature；与模型选择集不相交。
- 最终英文test=5,000、中文test=5,000。中文validation完全不使用。
- 初步实验中DANN超参数事先固定，不根据中文测试效果修改。
- 原始文件保留不变。输入预处理将空标题替换为空字符串、缺失类别替换为other；各原始划分的null字段计数保存在audit.json，所有文本字段在tokenization前验证类型。

## 四篇论文的对应关系

| Paper | 本次工作 | 可向老师使用的描述 |
|---|---|---|
| 1: Keung et al., 2020 | 相同数据集、mBERT、五星任务和English→Chinese设置；训练规模缩小 | MARC mBERT benchmark pilot / 部分基准复现 |
| 2: Conneau et al., 2020 | 使用公开XLM-R Base权重，在MARC微调；不训练CommonCrawl预训练任务 | XLM-R downstream application / 下游迁移实验 |
| 3: Ganin et al., 2016 | 实现GRL、语言域分类器、无标签目标数据训练；换用XLM-R与五星任务 | DANN method adaptation / 方法复现与跨语言扩展 |
| 6: Guo et al., 2017 | 实现标量Temperature Scaling及ECE/NLL/Brier，应用于MARC | Temperature Scaling method reproduction / 方法复现 |

不能写成“四篇论文所有原始结果均已完整复现”。XLM-R原论文不使用MARC；DANN原文为英语产品域/词袋二分类。本次组合是一个统一的pilot研究系统。

## 模型

AutoModel的最后一层CLS向量→dropout(0.1)→Linear(hidden,5)。所有模型使用相同分类头结构。

DANN另外使用GRL→Linear(hidden,256)→ReLU→dropout(0.1)→Linear(256,2)。域标签en=0、zh=1。

每步总损失=英文分类交叉熵+中英文域分类交叉熵。GRL强度=0.1×[2/(1+exp(-10p))-1]，p为训练进度。GRL只反转编码器收到的域损失梯度，域分类器正常最小化域损失。

XLM-R source-only与DANN均从同一基础权重及相同随机任务头初始化；DANN不从source-only最佳模型继续训练。两者在相同英文样本与相同轮数下比较。

## 校准与指标

Temperature T>0，通过最小化英文calibration集NLL拟合，log(T)边界对应T∈[0.01,100]。温度只缩放logits，不改变argmax和分类准确率。

- Accuracy与Macro-F1以0–1记录。
- MAE使用离散argmax星级，单位为星级，不是期望评分MAE。
- ECE为15个等宽置信度区间，按样本数加权的|accuracy−confidence|。
- NLL使用自然对数。
- Multiclass Brier为每个样本五类平方误差之和的均值，不再除以类别数。
- 校准器仅在英文拟合；中文校准改善不被预先假定。需要保留恶化结果。

## 与MARC原文的差异

原文使用200,000条训练样本、15轮、Adam、恒定8e-7学习率、batch32、body截断180wordpieces、MXNet/GluonNLP。本次为10,000条、2轮、AdamW、2e-5与warmup、FP16、总长度128、PyTorch。输入同样包含body/title/category。

原论文Table 2/3参考值：English supervised accuracy 63.3%、MAE 0.430；English→Chinese accuracy39.7%、MAE0.932。本次不同训练预算，不能把差值写成超越论文的证据。

## 限制

单种子结果无标准差，不能支持显著提升结论；后续至少3个种子。DANN接触目标无标签文本，source-only不接触，比较时必须明确。MARC语言之间产品类别分布不同，中文偏书籍，而且category字段本身为英文；不能将跨语言表现变化完全归因于语言。域分类准确率50%可能是表示坍塌或分类器弱，必须结合目标任务结果解释。FP16与CUDA内核可引入细微非确定性。

## 复核

运行`python -m unittest discover -s tests -v`检查梯度反转、已知指标值与校准优化。运行`python run.py verify`检查实际抽样与数据哈希、checkpoint哈希、所有测试集预测以及指标可从logits重算。
