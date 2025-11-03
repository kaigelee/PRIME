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



## 1. Introduction

 🔥 Pending


## 2. Environment Setup

For this project, we used python 3.10. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/PRIME-VAD
source ~/venv/PRIME-VAD/bin/activate
```

In that environment, the requirements can be installed with:

```shell
conda create --name lavad python=3.10
conda activate lavad
pip install -r requirements.txt
```


## 3. Dataset Setup


Please download the data, including captions, temporal summaries, indexes with their textual embeddings, and scores for the UCF-Crime and XD-Violence datasets, from the links below:

| Dataset     | Link                                                                                               |
| ----------- | -------------------------------------------------------------------------------------------------- |
| UCF-Crime   | [Google Drive](https://drive.google.com/file/d/1_7juCgOoWjQruyH3S8_FBqajuRaORmnV/view?usp=sharing) |
| XD-Violence | [Google Drive](https://drive.google.com/file/d/1yzDP1lVwPlA_BS2N5Byr1PcaazBklfkI/view?usp=sharing) |

and place them in the `/path/to/directory/datasets` folder. The code works with pre-extracted frames, which are not provided within the zipped files. You can download the videos from the official websites ([UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) and [XD-Violence](https://roc-ng.github.io/XD-Violence/)) and extract the frames yourself using a script similar to `scripts/00_extract_frames.sh`. Please note that you need to change the paths within the files included in the `annotations` folder of each dataset accordingly.


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
    Implementation of Pareto-based candidate selection (Section 1-86 of the paper) adapted for binary metrics
    Input:
        P: Candidate pool, list where each element is a candidate prompt (string describing core logic of the prompt)
        SP: Binary result matrix, numpy array with shape=(len(P), len(instance set)), SP[k][i] ∈ {0,1} (0=incorrect, 1=correct)
    Output:
        selected_idx: Index of the sampled candidate in P
    """
    # 1. Construct optimal candidate set P*[i] for each instance i (Steps 2-5 in Section 1-86 of the paper)
    num_instances = SP.shape[1]  # Total number of instances
    P_star = [[] for _ in range(num_instances)]  # P_star[i] is the optimal candidate set for instance i
    for i in range(num_instances):
        # For instance i, find indices of all candidates with correct predictions (SP[k][i] == 1)
        best_candidates_idx = np.where(SP[:, i] == 1)[0].tolist()
        # If all candidates are incorrect, retain all candidates (to avoid selection interruption, per candidate retention logic in Section 1-73 of the paper)
        if not best_candidates_idx:
            best_candidates_idx = list(range(len(P)))
        P_star[i] = [P[k] for k in best_candidates_idx]  # Store candidate objects
    
    # 2. Integrate global candidate pool C (Steps 6-7 in Section 1-86: remove duplicates)
    C = []
    for candidates in P_star:
        for cand in candidates:
            if cand not in C:
                C.append(cand)
    # Remove candidates that are "incorrect on all instances" (implicit logic in Section 1-86: retain valid strategies)
    valid_candidates = []
    for cand in C:
        cand_idx = P.index(cand)
        if np.any(SP[cand_idx] == 1):
            valid_candidates.append(cand)
    C = valid_candidates
    if not C:
        return 0  # Extreme case: no valid candidates, return initial candidate
    
    # 3. Remove strictly dominated candidates (Steps 8-13 in Section 1-86)
    D = []  # Store dominated candidates
    for idx_x in range(len(C)):
        x = C[idx_x]
        x_idx = P.index(x)  # Index of x in original candidate pool P
        x_correct = set(np.where(SP[x_idx] == 1)[0].tolist())  # Set of instances where x is correct
        # Check if x is dominated by other candidates
        is_dominated = False
        for idx_y in range(len(C)):
            if idx_x == idx_y:
                continue
            y = C[idx_y]
            y_idx = P.index(y)
            y_correct = set(np.where(SP[y_idx] == 1)[0].tolist())  # Set of instances where y is correct
            # Domination condition: x's correct set is a subset of y's, and y has correct instances not covered by x (adapted from domination definition in Section 1-86)
            if x_correct.issubset(y_correct) and len(y_correct - x_correct) > 0:
                is_dominated = True
                break
        if is_dominated:
            D.append(x)
    # Prune candidate pool: remove dominated candidates to get Ĉ (Step 13 in Section 1-86)
    C_hat = [cand for cand in C if cand not in D]
    if len(C_hat) == 1:
        return P.index(C_hat[0])  # Only 1 candidate, return directly
    
    # 4. Sample with probability proportional to f[Φk] (Steps 14-16 in Section 1-86: f[Φk] is the number of instances where the candidate enters P*[i])
    f = []
    for cand in C_hat:
        cand_idx = P.index(cand)
        # f[Φk] = number of instances where the candidate is correct (i.e., number of instances where it enters P*[i])
        f_k = np.sum(SP[cand_idx] == 1)
        f.append(f_k)
    # Sample proportionally to f[k] (e.g., weighted sampling)
    total_f = sum(f)
    probabilities = [fk / total_f for fk in f]
    selected_cand = np.random.choice(C_hat, p=probabilities)
    return P.index(selected_cand)

# ------------------------------ Example Input ------------------------------
# 1. Candidate pool P: 3 prompt candidates for video anomaly detection (describing core logic, corresponding to A/B/C above)
P = [
    # Candidate A: Focus only on limb posture changes for behavioral anomalies
    "When extracting abnormal features, focus solely on limb posture changes of pedestrians in consecutive frames to determine behavioral anomalies",
    # Candidate B: Focus on key features of behavioral + object anomalies
    "When extracting abnormal features, cover pedestrian limb posture changes and object structural integrity to determine behavioral/object anomalies",
    # Candidate C: Cover comprehensive features of behavioral + object + environmental anomalies
    "When extracting abnormal features, include limb posture, object structure, and environmental light intensity/texture changes to determine all types of anomalies"
]

# 2. Binary result matrix SP: shape=(3 candidates, 5 instances), where SP[k][i] indicates the prediction result of candidate k on instance i (1=correct, 0=incorrect)
# Instance order: i=0 (behavioral-fall), i=1 (behavioral-running), i=2 (object-shelf collapse), i=3 (environmental-sudden dimming), i=4 (environmental-floor water)
SP = np.array([
    [1, 0, 0, 0, 0],  # Candidate A: Correct only on instance 0
    [1, 1, 1, 0, 0],  # Candidate B: Correct on instances 0, 1, 2
    [1, 1, 1, 1, 1]   # Candidate C: Correct on all instances
])

# ------------------------------ Function Call & Output ------------------------------
np.random.seed(42)  # Fix random seed for reproducibility
selected_idx = SELECTCANDIDATE(P, SP)
print(f"Selected candidate index: {selected_idx}")
print(f"Selected candidate prompt: {P[selected_idx]}")
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
