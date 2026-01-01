# <p align=center>`PRIME for Video Anomaly Detection`</p><!-- omit in toc -->

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

 🔥 Pending


🔑 **Key Idea**


**Pareto Front-based candidate selection Training Pseudocode**

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

PRIME is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [VERA](https://github.com/vera-framework/VERA/tree/main)
* [LAVAD ](https://github.com/lucazanella/lavad/tree/main)



## Code Availability Statement
This code is associated with a paper currently under review. To comply with the review process, the code will be made FULLY available once the paper is accepted.  :smiley:

We appreciate your understanding and patience. Once the code is released, we will warmly welcome any feedback and suggestions. Please stay tuned for our updates!
