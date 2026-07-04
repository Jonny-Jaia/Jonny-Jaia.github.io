# CV 基础模型：ViT / CLIP / MAE / DINO / SAM / Diffusion

## 当前定位

这一章用于补齐大模型算法工程师的视觉基础模型底座。面试里如果只会说“多模态就是图像 encoder 接 LLM”，会显得很薄；更好的回答是先把视觉基础模型分成三条线：**表征学习**、**开放词表语义对齐**、**生成建模**。

```mermaid chapter-map
Image pixels -> Visual tokens -> Representation / Alignment / Generation -> VLM or CV downstream tasks
```

| 方向 | 代表模型 | 核心问题 | 和大模型岗位的关系 |
|---|---|---|---|
| 视觉 Transformer | ViT、Swin | 图像如何 token 化并进入 Transformer | VLM 视觉 encoder 的主流骨架 |
| 图文对齐 | CLIP、ALIGN | 如何把图像和文本放到同一语义空间 | 检索、zero-shot、视觉指令模型的语义底座 |
| 自监督视觉 | MAE、DINO、DINOv2 | 无需人工标签学习高质量视觉表征 | 视觉预训练、低标注迁移、通用视觉特征 |
| 检测与分割 | DETR、Grounding DINO、SAM | 如何定位物体、区域和 mask | GUI Agent、视觉工具、grounding、多模态 RAG |
| 生成模型 | DDPM、Latent Diffusion、Stable Diffusion | 如何从噪声生成可控图像 | 文生图、图像编辑、多模态数据合成 |

## ViT：图像如何进入 Transformer

ViT 的核心是把图像切成 patch，再把每个 patch 线性投影成 token。若图像大小为 $H \times W$，patch 大小为 $P \times P$，token 数为：

$$
N = \frac{H W}{P^2}
$$

每个 patch 被展平为 $P^2C$ 维向量，再通过线性层映射到 Transformer hidden size。随后加上位置编码，送入标准 Transformer encoder。

```mermaid chapter-map
Image -> Patchify -> Linear projection -> Positional embedding -> Transformer encoder -> CLS or pooled feature
```

### ViT 与 CNN 的区别

| 维度 | CNN | ViT |
|---|---|---|
| 归纳偏置 | 局部性、平移等变、层级特征 | 归纳偏置弱，更依赖大数据预训练 |
| 感受野 | 层层扩大 | self-attention 天然全局交互 |
| 数据效率 | 中小数据上更稳 | 大数据预训练后迁移能力强 |
| 可扩展性 | 卷积结构强，扩展有上限 | 和 Transformer scaling 规律更接近 |

**面试结论**：ViT 不是“CNN 完全过时”，而是当数据规模和模型规模足够大时，弱归纳偏置 + 全局 attention + 统一 Transformer 架构更适合做通用视觉底座。

## CLIP：图文对齐的基础范式

CLIP 使用双塔结构：image encoder 编码图像，text encoder 编码文本，然后用 batch 内图文对比学习把匹配图文拉近，不匹配图文推远。

```mermaid chapter-map
Image encoder -> image embedding -> contrastive loss <- text embedding <- Text encoder
```

设一批样本中图像 embedding 为 $I_i$，文本 embedding 为 $T_i$，相似度矩阵为：

$$
s_{ij}=\frac{I_i^\top T_j}{\tau}
$$

图到文损失为：

$$
\mathcal{L}_{i2t}=-\frac{1}{B}\sum_i \log \frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}
$$

文到图损失同理，最终通常取二者平均。

### CLIP 为什么能 zero-shot

CLIP 训练时不是固定类别分类，而是学习“图像和自然语言描述是否匹配”。推理时可以把类别名写成 prompt，例如 “a photo of a dog”，将图像 embedding 与各类别文本 embedding 比较，相似度最高的就是预测类别。

### CLIP 的局限

- 它擅长全局语义对齐，但不天然擅长精细定位、计数、OCR、空间关系。
- 对 prompt 敏感，类别文本模板会影响 zero-shot 结果。
- 图文对齐来自大规模网络数据，可能继承偏见、噪声和长尾覆盖不足。
- CLIP embedding 适合检索和粗粒度语义，但不能直接替代 VLM 的多轮视觉推理。

## MAE：视觉自监督的 masked reconstruction

MAE 的核心是随机 mask 大比例图像 patch，只把可见 patch 输入 encoder，再用轻量 decoder 重建被 mask 的像素或 patch。视觉图像有强空间冗余，因此高 mask ratio 仍能形成有效学习任务。

```mermaid chapter-map
Image patches -> Random mask -> Encode visible patches -> Lightweight decoder -> Reconstruct masked patches
```

| 设计 | 作用 |
|---|---|
| 高比例 mask | 增加任务难度，避免简单复制局部纹理 |
| 非对称 encoder-decoder | encoder 只处理可见 patch，训练更高效 |
| pixel reconstruction | 直接重建低层信号，简单稳定 |

**面试结论**：MAE 类方法说明视觉自监督不一定要依赖文本，可以通过“遮挡后恢复”学习可迁移表征；但重建目标偏低层，和 CLIP 的语义对齐目标互补。

## DINO / DINOv2：自蒸馏视觉表征

DINO 系列使用 teacher-student 自蒸馏：同一图像的不同增强视图分别送入 student 和 teacher，student 学习匹配 teacher 的输出分布。teacher 通常由 student 的 EMA 更新得到。

```mermaid chapter-map
Image augmentations -> Student / Teacher -> Distribution matching -> Strong visual representation
```

DINO 类表征常被认为有较强的语义分割、物体区域和局部结构感知能力。它和 CLIP 的区别是：CLIP 更强调图文语义对齐，DINO 更强调纯视觉自监督表征。

## DETR / Grounding DINO / SAM：从识别到定位与分割

### DETR

DETR 把目标检测看成 set prediction：固定数量 object queries 通过 Transformer decoder 与图像特征交互，输出一组 box/class，再用 Hungarian matching 和真实目标做一一匹配。

**面试抓手**：DETR 的核心不是“用了 Transformer”，而是把检测从 anchor/NMS 风格改成集合预测，训练目标也随之变成 bipartite matching。

### Grounding DINO

Grounding DINO 解决 open-set detection：输入文本类别或短语，让模型定位图像中对应区域。它把 DINO/DETR 类检测能力和语言 grounding 结合，适合做“文本指令 -> 视觉区域”的工具。

### SAM

SAM 是 promptable segmentation：输入 point、box、mask 或文本/其他提示形式，输出分割 mask。它的关键价值是把分割从固定类别任务变成可提示的基础能力。

```mermaid chapter-map
Image encoder -> Prompt encoder -> Mask decoder -> Segmentation masks
```

| 模型 | 输出 | 与 VLM/Agent 的关系 |
|---|---|---|
| DETR | box + class | 结构化定位，适合检测基础 |
| Grounding DINO | text-conditioned box | 根据语言找区域，适合视觉工具调用 |
| SAM | mask | 精细区域分割，适合视觉编辑、GUI 元素分割、图像理解工具 |

## DDPM：扩散模型的基本思想

扩散模型由 forward noising 和 reverse denoising 组成。

前向过程逐步往真实图像 $x_0$ 加高斯噪声：

$$
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

训练时常用噪声预测目标：

$$
\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]
$$

推理时从纯噪声开始，逐步去噪得到图像。

### 扩散模型为什么稳定

GAN 训练是生成器和判别器的博弈，容易 mode collapse；DDPM 的训练更像有监督噪声预测，每个时间步都有明确的 regression target，因此训练更稳定，但推理通常需要多步去噪，速度更慢。

## Latent Diffusion：为什么进入 latent 空间

像素空间扩散在高分辨率图像上计算成本高。Latent Diffusion 先用 VAE 把图像压到 latent 空间，再在 latent 上做扩散，最后 decoder 还原图像。

```mermaid chapter-map
Image -> VAE encoder -> Latent diffusion -> VAE decoder -> Image
```

它还通过 cross-attention 接入文本条件，使文本、mask、box 等条件可以控制生成过程。

**面试结论**：Latent Diffusion 的核心 trade-off 是用压缩 latent 降低计算成本，同时尽量保留视觉细节；这也是 Stable Diffusion 系列能够较低成本做高分辨率生成的重要原因。

## 与多模态 LLM 的连接

| VLM 组件 | 常见选择 | 作用 |
|---|---|---|
| 视觉 encoder | ViT、CLIP ViT、DINOv2、SigLIP | 把图像转成视觉 token 或 embedding |
| connector | Linear、MLP projector、Q-Former、Perceiver Resampler | 把视觉特征映射到 LLM hidden space 或压缩成少量 token |
| LLM backbone | LLaMA/Qwen/GLM/InternLM 等 | 负责语言推理、指令遵循和答案生成 |
| 训练阶段 | 图文对齐、视觉指令微调、多模态 RLHF/RLVR | 从看懂图到按指令推理，再到偏好和任务优化 |
| 工具模块 | OCR、Grounding DINO、SAM、检测器、检索器 | 弥补 VLM 在精细定位、OCR、区域操作上的短板 |

## 原理代码：CV 基础模型最小实现

关联原理代码：

- [ViT / CLIP / MAE / DDPM kernels](#principle-code/cv-foundation-kernels)：直接展示 `vit_patchify`、`clip_contrastive_loss`、`mae_random_mask`、`diffusion_q_sample` 四个可运行函数。

面试时可以按这四个函数组织回答：

| 函数 | 对应知识点 | 面试抓手 |
|---|---|---|
| `vit_patchify` | ViT 图像 token 化 | token 数量随 patch size 平方反比变化，patch 越小计算越贵但细节越多。 |
| `clip_contrastive_loss` | CLIP 图文对比学习 | batch 内正负样本构造、双向 CE、temperature 控制分布尖锐度。 |
| `mae_random_mask` | MAE masked image modeling | encoder 只处理可见 patch，decoder 负责按原顺序重建，训练效率更高。 |
| `diffusion_q_sample` | DDPM 前向加噪 | 由 $x_0$ 和噪声 $\epsilon$ 闭式采样 $x_t$，训练通常预测噪声。 |
## 面试 QA

**Q：ViT 和 CNN 最大区别是什么？**

A：CNN 有强局部归纳偏置，适合中小数据和局部纹理；ViT 把图像 patch 化后用 self-attention 做全局交互，归纳偏置弱但可扩展性强，适合大规模预训练和迁移。

**Q：CLIP 为什么能做 zero-shot？**

A：它把图像和自然语言编码到同一个语义空间。分类时把类别写成文本 prompt，比较图像 embedding 与类别文本 embedding 的相似度，不需要为每个新类别单独训练分类头。

**Q：MAE 和 CLIP 都是预训练，差别是什么？**

A：MAE 是视觉自监督重建，主要从图像内部结构学习表征；CLIP 是图文对比学习，从自然语言监督学习开放语义。MAE 更偏视觉结构，CLIP 更偏语义对齐。

**Q：SAM 为什么叫 foundation model？**

A：因为它不是只分固定类别，而是通过 point/box/mask 等 prompt 输出 mask，能 zero-shot 迁移到多种分割场景。它提供的是“可提示分割能力”，类似视觉工具底座。

**Q：扩散模型和自回归生成有什么区别？**

A：自回归通常按 token 顺序生成；扩散模型从噪声出发逐步去噪，每一步都修正整幅图或 latent。扩散更适合连续视觉空间建模，但推理步数和调度器会影响速度。

**Q：为什么多模态大模型还需要 OCR、SAM、Grounding DINO 这类工具？**

A：VLM 的视觉 encoder 通常会压缩图像信息，精细文字、区域边界、坐标关系可能丢失。OCR、检测、分割工具能提供结构化证据，降低视觉幻觉，提升可解释性和可控性。

## 后续拓展

- 补 ConvNeXt、Swin、EVA、SigLIP 的视觉 encoder 对比。
- 补 VQ-VAE / VQGAN / tokenizer 在图像生成中的作用。
- 补视频基础模型：TimeSformer、VideoMAE、InternVideo、Video-LLaVA。
- 补视觉评测：MMBench、MMMU、OCRBench、TextVQA、RefCOCO、SEED-Bench。

## 知识索引引用

| 知识点 | 来源 | 本页使用方式 |
|---|---|---|
| ViT patchify 与纯 Transformer 图像分类 | https://arxiv.org/abs/2010.11929 | 解释图像 token 化、ViT 与 CNN 差异 |
| CLIP 图文对比学习与 zero-shot | https://arxiv.org/abs/2103.00020 | 解释双塔对比学习、开放词表分类与局限 |
| MAE masked reconstruction | https://arxiv.org/abs/2111.06377 | 解释视觉自监督、非对称 encoder-decoder 与高 mask ratio |
| DDPM 噪声预测目标 | https://arxiv.org/abs/2006.11239 | 解释 forward noising、reverse denoising 和稳定训练 |
| Latent Diffusion / Stable Diffusion 基础 | https://arxiv.org/abs/2112.10752 | 解释 latent 空间扩散、cross-attention 条件控制 |
| SAM 可提示分割 | https://arxiv.org/abs/2304.02643 | 解释 promptable segmentation 与视觉工具底座 |
| Grounding DINO 开放集检测 | https://arxiv.org/abs/2303.05499 | 解释 language-conditioned detection 与 grounding |
