# <p align=center>`PRIME for Video Anomaly Detection`</p><!-- omit in toc -->

Kaige Li, Weiming Shi, and Xiaochun Cao*, IEEE Senior Member

*Corresponding author: [Xiaochun Cao](https://scholar.google.com/citations?user=PDgp6OkAAAAJ&hl=en).

## Table of Contents

  * [Introduction](#1-introduction)
  * [Environment Setup](#2-Environment-Setup)
  * [Dataset](#3-Dataset-Setup)
  * [Framework Structure](#4-Framework-Structure)
  * [Acknowledgements](#5-Acknowledgements)
  * [Future Work](#6-future-work)


## Code Implementation Statement

As discussed in [8](https://github.com/lhoyer/MIC/issues/8), [54](https://github.com/lhoyer/MIC/issues/54) and [63](https://github.com/lhoyer/MIC/issues/63), **our method inherits the instability of [MIC](https://github.com/lhoyer/MIC).** :cry:

Note, however, that the mathematical expectation of performance is the same for both, i.e., **76.9%** mIoU and **69.9%** mIoU on GTAV→Cityscapes and SYNTHIA→Cityscapes, respectively. :100:

## 1. Introduction

 🔥 Pending


## 2. Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/LVP-UDASeg
source ~/venv/LVP-UDASeg/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

## 3. Dataset Setup

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.


The final folder structure should look like this:

```none
LVP
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## 4. Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).


🔑 **Key Idea**

Our Language-Vision Prior (LVP) combines:

* Language Prior (LP): multi-prototype prompts capture class-level semantics and intra-class variance.

* Vision Prior (VP): bi-directional masking encourages robust global-local reasoning.

Together, they guide stable and reliable domain adaptation.

**Overall Training Pseudocode**

```python
import numpy as np

def SELECTCANDIDATE(P, SP):
    """
    论文1-86节Pareto-based candidate selection的二值指标适配实现
    Input:
        P: 候选池，列表，每个元素为候选提示（字符串形式，描述提示核心逻辑）
        SP: 二值结果矩阵，numpy数组，shape=(len(P), len(实例集))，SP[k][i] ∈ {0,1}（0=错误，1=正确）
    Output:
        selected_idx: 抽样选中的候选在P中的索引
    """
    # 1. 构建每个实例i的最优候选集合P*[i]（论文1-86节步骤2-5）
    num_instances = SP.shape[1]  # 实例总数
    P_star = [[] for _ in range(num_instances)]  # P_star[i]为实例i的最优候选集合
    for i in range(num_instances):
        # 对实例i，找到所有预测正确（SP[k][i]==1）的候选索引k
        best_candidates_idx = np.where(SP[:, i] == 1)[0].tolist()
        # 若所有候选均错误，保留全部候选（避免筛选中断，论文1-73节候选池保留逻辑）
        if not best_candidates_idx:
            best_candidates_idx = list(range(len(P)))
        P_star[i] = [P[k] for k in best_candidates_idx]  # 存储候选对象
    
    # 2. 整合全局候选池C（论文1-86节步骤6-7：去重）
    C = []
    for candidates in P_star:
        for cand in candidates:
            if cand not in C:
                C.append(cand)
    # 剔除“在所有实例上均错误”的候选（论文1-86节隐含逻辑：保留有效策略）
    valid_candidates = []
    for cand in C:
        cand_idx = P.index(cand)
        if np.any(SP[cand_idx] == 1):
            valid_candidates.append(cand)
    C = valid_candidates
    if not C:
        return 0  # 极端情况：无有效候选，返回初始候选
    
    # 3. 剔除严格支配候选（论文1-86节步骤8-13）
    D = []  # 存储被支配的候选
    for idx_x in range(len(C)):
        x = C[idx_x]
        x_idx = P.index(x)  # x在原始候选池P中的索引
        x_correct = set(np.where(SP[x_idx] == 1)[0].tolist())  # x的正确实例集合
        # 检查x是否被其他候选支配
        is_dominated = False
        for idx_y in range(len(C)):
            if idx_x == idx_y:
                continue
            y = C[idx_y]
            y_idx = P.index(y)
            y_correct = set(np.where(SP[y_idx] == 1)[0].tolist())  # y的正确实例集合
            # 支配条件：x的正确集合被y包含，且y有x未覆盖的正确实例（论文1-86节支配定义适配）
            if x_correct.issubset(y_correct) and len(y_correct - x_correct) > 0:
                is_dominated = True
                break
        if is_dominated:
            D.append(x)
    # 修剪候选池：移除被支配候选，得到ˆC（论文1-86节步骤13）
    C_hat = [cand for cand in C if cand not in D]
    if len(C_hat) == 1:
        return P.index(C_hat[0])  # 仅1个候选，直接返回
    
    # 4. 按f[Φk]概率抽样（论文1-86节步骤14-16：f[Φk]为候选进入P*[i]的实例数）
    f = []
    for cand in C_hat:
        cand_idx = P.index(cand)
        # f[Φk] = 候选在所有实例上正确的数量（即进入P*[i]的实例数）
        f_k = np.sum(SP[cand_idx] == 1)
        f.append(f_k)
    # 按f[k]正比抽样（如权重抽样）
    total_f = sum(f)
    probabilities = [fk / total_f for fk in f]
    selected_cand = np.random.choice(C_hat, p=probabilities)
    return P.index(selected_cand)

# ------------------------------ 示例输入 ------------------------------
# 1. 候选池P：3个视频异常检测的提示候选（描述核心逻辑，对应前文A/B/C）
P = [
    # 候选A：仅关注行为异常的肢体姿态变化
    "提取异常特征时，仅聚焦连续帧中行人的肢体姿态变化，判定行为异常",
    # 候选B：关注行为+物体异常的关键特征
    "提取异常特征时，覆盖行人肢体姿态变化、物体结构完整性，判定行为/物体异常",
    # 候选C：覆盖行为+物体+环境异常的全面特征
    "提取异常特征时，包含肢体姿态、物体结构、环境光强/纹理变化，判定全类别异常"
]

# 2. 二值结果矩阵SP：shape=(3个候选, 5个实例)，SP[k][i]表示候选k在实例i上的预测结果（1=正确，0=错误）
# 实例顺序：i=0(行为-跌倒)、i=1(行为-奔跑)、i=2(物体-货架坍塌)、i=3(环境-灯光骤暗)、i=4(环境-地面积水)
SP = np.array([
    [1, 0, 0, 0, 0],  # 候选A：仅正确实例0
    [1, 1, 1, 0, 0],  # 候选B：正确实例0、1、2
    [1, 1, 1, 1, 1]   # 候选C：正确所有实例
])

# ------------------------------ 函数调用与输出 ------------------------------
np.random.seed(42)  # 固定随机种子，确保结果可复现
selected_idx = SELECTCANDIDATE(P, SP)
print(f"选中的候选索引：{selected_idx}")
print(f"选中的候选提示：{P[selected_idx]}")
```

##  5. Acknowledgements

TIP is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)

##  6. Future Work


## Multi-Prototype Representation

Current results indicate that **small-object classes** (e.g., *traffic light*, *traffic sign*, *pole*) show higher intra-class diversity, while **large-area classes** (e.g., *road*, *sky*) appear more homogeneous. Using a single prototype per class may not be sufficient to capture such diversity.

### Directions

- **Adaptive Prototype Allocation**
  - Allocate prototypes per class based on:
    - *Intra-class diversity* (e.g., covariance trace, mean pairwise distance).
    - *Effective sample size* (e.g., log of pixel count).
    - *Resource budget* (global prototype limit with min/max constraints).

- **Dynamic Selection**
  - Explore automatic methods to determine prototype counts:
    - *k-means* with silhouette or Davies–Bouldin scores.
    - *Gaussian Mixture Models* with BIC/AIC.

- **Class-Specific Strategies**
  - Small-object classes with heterogeneous appearance → more prototypes.
  - Large-object classes with stable texture → fewer prototypes.

- **Evaluation Metrics**
  - Monitor **intra-class coverage** (distance to nearest prototype).
  - Monitor **inter-class separation** (margin to non-class prototypes).
  - Use these signals to refine prototype allocation.

---

*The goal is to better capture intra-class variability without overspending resources, paving the way for finer-grained representation and improved segmentation quality.*



## Prototype Allocation

This repository provides a utility function to allocate prototype counts per class  
based on intra-class diversity and sample size.

## Example: Allocate Prototypes

```python

import torch
import math

def allocate_prototypes(feats_by_class, K_total, K_min=1, K_max=10, alpha=0.7, beta=0.3, eps=1e-8):
    """
    Allocate prototype counts per class based on intra-class diversity and sample size.

    Args:
        feats_by_class (dict[int, torch.Tensor]): A dictionary mapping class -> features (N_c, C).
        K_total (int): Total number of prototypes across all classes.
        K_min (int): Minimum number of prototypes per class (default=1).
        K_max (int): Maximum number of prototypes per class (default=10).
        alpha (float): Weight for diversity in allocation (default=0.7).
        beta (float): Weight for sample count in allocation (default=0.3).
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        dict[int, int]: A dictionary mapping each class to its allocated number of prototypes.
    """
    classes = sorted(feats_by_class.keys())
    D, L = [], []  # store diversity and log-count values

    for c in classes:
        X = feats_by_class[c]
        # Use covariance trace as a measure of diversity
        Xc = X - X.mean(dim=0, keepdim=True)
        cov_trace = (Xc.T @ Xc / max(1, X.shape[0]-1)).diag().sum().item()
        D.append(max(cov_trace, 0.0))
        L.append(math.log1p(X.shape[0]))  # log(1 + sample size)

    # Normalize diversity and sample size contributions
    D_sum = sum(D) + eps
    L_sum = sum(L) + eps
    d_hat = [d / D_sum for d in D]
    n_hat = [l / L_sum for l in L]

    # Initial allocation: ensure each class has at least K_min
    base = K_min * len(classes)
    room = max(K_total - base, 0)
    q = [alpha * d + beta * n for d, n in zip(d_hat, n_hat)]
    q_sum = sum(q) + eps
    k_float = [K_min + room * (qi / q_sum) for qi in q]  # float allocation

    # Round allocations and apply min/max limits
    k_round = [int(round(x)) for x in k_float]
    k_round = [max(K_min, min(K_max, k)) for k in k_round]

    # Adjust to make sure the total sum equals K_total
    diff = K_total - sum(k_round)
    if diff != 0:
        # Priority: adjust classes whose rounded value deviates most from float target
        prio = sorted(
            range(len(classes)),
            key=lambda i: (k_float[i] - k_round[i]),
            reverse=(diff > 0),
        )
        i = 0
        while diff != 0 and i < len(prio):
            idx = prio[i]
            newk = k_round[idx] + (1 if diff > 0 else -1)
            if K_min <= newk <= K_max:
                k_round[idx] = newk
                diff += -1 if diff > 0 else 1
            i += 1

    K_dict = {c: k for c, k in zip(classes, k_round)}
    return K_dict

```




## Code Availability Statement
This code is associated with a paper currently under review. To comply with the review process, the code will be made FULLY available once the paper is accepted.  :smiley:

We appreciate your understanding and patience. Once the code is released, we will warmly welcome any feedback and suggestions. Please stay tuned for our updates!
