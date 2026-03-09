# 面向当前 UReader 代码库的改造行动清单（Shape-Adaptive Cropping + 可插拔 LoRA + 视觉上下文位置编码）

> 目标：在**尽量复用现有工程骨架**的前提下，把你提出的 3 个创新点改造成可训练、可评估、可部署的版本。

---

## 0. 先做一次“现状盘点”（1~2 天）

### 你现在已经具备的能力
1. **形状相关的多尺度切片已经存在原型**：
   - `DocPretrainProcessor`/`DocNewMultiScaleSFTProcessor` 使用 `AnchorResize` 根据宽高比选择 anchor，再切成 patch 序列。  
   - 已输出 `patch_position` 并向后传递。  
2. **LoRA 已在训练脚本中接入**：
   - `pipeline/train.py` 已通过 PEFT 的 `LoraConfig + get_peft_model` 对语言侧做 LoRA。  
3. **位置感知机制已有“可插位”设计**：
   - `patch_positions` 已进入 visual abstractor，且支持 `patch_pos_embed_type in {pre, post}` 的注入方式。  

### 主要短板（与你提案对齐）
1. 现有裁剪主要依据 **宽高比 anchor**，并非“内容密度驱动 ROI 提议网络”。
2. LoRA 目前偏向语言侧默认路径，尚未形成“按任务热插拔 + 多模块细粒度 target_modules 管控 + 导出规范”。
3. 位置编码仍以 patch 索引嵌入为主，缺少你提到的“2D 连续坐标 + 相对偏置 + layout prior”的统一接口。

---

## 1. 创新点一：Shape-Adaptive Cropping（先从“无监督内容密度版本”起步）

### 1.1 第一阶段（低风险、快速验证）
**目标**：在不引入重型检测器的前提下，把裁剪从“纯宽高比”升级到“内容密度 + 宽高比联合”。

- 在 `pipeline/data_utils/processors/doc_processor.py` 新增一个处理器（例如 `DocAdaptiveMultiScaleSFTProcessor`）：
  1. 对图像构建简易 saliency map（可选：边缘、局部方差、Sobel/Scharr 响应叠加）；
  2. 先生成候选 ROI（滑窗 + NMS 或阈值连通域）；
  3. 对每个 ROI 做 **aspect ratio preserving** 缩放填充至 patch 尺寸；
  4. 输出 `(N, C, H, W)` 与 `patch_positions`，兼容现有 batch 流。

- 在 `configs/sft/release.yaml` 新增 processor 配置开关，允许 A/B：
  - `DocNewMultiScaleSFTProcessor`（baseline）
  - `DocAdaptiveMultiScaleSFTProcessor`（new）

- 在 `pipeline/data_utils/xgpt3_dataset.py` 保持调用路径不变（避免改 dataset 主逻辑），只换 processor 实现。

### 1.2 第二阶段（高收益）
**目标**：引入轻量可学习 proposal head。

- 建议在视觉前处理旁路增加一个小 CNN head 预测 ROI（可先离线训练/蒸馏，再联训）。
- 初版不必端到端硬耦合主干，先做“离线 proposal + 在线训练主干”的两阶段实验，降低工程复杂度。

### 1.3 验收指标
- 长文档/表格任务上，统计：
  - 结构化抽取 F1、ANLS、Exact Match；
  - 推理时 token 数、平均 patch 数、吞吐。
- 与当前 anchor-only 方案做对照。

---

## 2. 创新点二：可插拔 LoRA（从“能训”升级到“可运维”）

### 2.1 你当前代码可直接利用的部分
- `pipeline/train.py` 已有 `--language-training-method lora` 和 `target_modules` 正则注入。

### 2.2 需要补齐的工程化能力
1. **模块粒度化开关**：
   - 拆分配置项：`--lora-on-language`, `--lora-on-abstractor`, `--lora-on-vision`；
   - 每个模块可独立配置 `r/alpha/dropout/target_modules`。
2. **可插拔装载协议**：
   - 推理入口 `pipeline/interface.py` 增加 `adapter_path` 与 `adapter_name`；
   - 支持 runtime load/unload/switch（多租户场景关键）。
3. **训练导出规范**：
   - 基座权重与 adapter 权重分离保存；
   - 记录 adapter 元数据（任务、数据版本、r、target_modules、基座 commit hash）。

### 2.3 推荐实施顺序
- 先只做 language + abstractor 两处 LoRA；
- vision LoRA 作为可选实验项（收益依赖数据规模）。

---

## 3. 创新点三：视觉上下文感知位置编码（分三层推进）

### 3.1 第一层：连续二维坐标注入（最小改动）
- 在现有 `patch_positions` 基础上，构建归一化坐标 `(x, y) in [0,1]`；
- 在 `mplug_owl/modeling_mplug_owl.py` 的 patch embedding 注入点（`pre/post`）新增 `coord_mlp` 或 embedding 映射后相加。

### 3.2 第二层：相对位置偏置
- 在视觉自注意力中增加相对偏置项（可先离散桶化坐标差）；
- 保持可配置开关，便于回退。

### 3.3 第三层：layout prior（软约束）
- 加入先验 token 或辅助损失（例如标题区概率、表格区概率）作为弱监督；
- 建议先在数据侧生成 pseudo layout 标签，避免一次性改模型过深。

---

## 4. 建议的分里程碑排期

### M1（1 周）：可运行基线增强
- 新增 `DocAdaptiveMultiScaleSFTProcessor`（非学习版 ROI）；
- 配置开关接入 + 训练可跑 + 快速评估脚本可对比。

### M2（1 周）：LoRA 可插拔化
- 训练脚本模块化 LoRA 开关；
- 推理接口支持 adapter 动态加载；
- 输出统一 adapter 目录结构。

### M3（1~2 周）：位置编码增强
- 连续坐标注入 + 相对偏置开关；
- 小规模 ablation（pre/post × with/without relative bias）。

### M4（1 周）：联调与消融
- 3 大改动组合实验；
- 对照 Pix2Struct/Donut 风格固定分辨率策略，输出报告。

---

## 5. 具体改动落点（按文件）

1. **裁剪模块**：
   - `pipeline/data_utils/processors/doc_processor.py`
   - `configs/sft/release.yaml`
2. **数据流与批处理兼容**：
   - `pipeline/data_utils/xgpt3_dataset.py`
   - `pipeline/utils.py`
3. **位置编码与视觉抽象器**：
   - `mplug_owl/modeling_mplug_owl.py`
4. **LoRA 训练/推理可插拔**：
   - `pipeline/train.py`
   - `pipeline/interface.py`

---

## 6. 风险与规避建议

1. **patch 数激增导致显存暴涨**
   - 设 `max_patches_per_image` + 动态裁剪置信度阈值。
2. **ROI 偏差导致语义缺失**
   - 保留全图 token（你方案里的 global + local 双路非常必要）。
3. **多改动同时上线难定位收益**
   - 严格做单因素消融：
     - baseline
     - +adaptive crop
     - +LoRA plugin
     - +pos encoding
     - full

---

## 7. 你现在就可以执行的“第一批命令级动作”

1. 新建 processor 与配置开关，先跑 1k 样本 smoke train。  
2. 固化评估集，输出与当前 release 配置的差异表。  
3. 再进入 LoRA 插拔接口改造，最后做位置编码增强。  

> 这样做的好处：你可以在第 1 周就拿到可量化收益，避免陷入一次性重构风险。
