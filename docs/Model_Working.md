# Model Working: Transformer-based QST with Classical Shadows

## Overview
This implementation follows **Track 1** of Assignment 2, utilizing a **Transformer** architecture to reconstruct 2-qubit density matrices from Classical Shadows.

## 1. Classical Shadows
For each qubit, we perform a randomized Pauli measurement ($X, Y,$ or $Z$) and record the outcome ($0$ or $1$). This produces a "shadow" snapshot. A collection of $K=50$ shadows is used per state.

## 2. Transformer Architecture
Unlike simple MLPs, the **Transformer Encoder** is designed to process sequences of data while capturing the correlations between individual shadow samples through **Self-Attention**.

- **Input Projection**: The basis indices and outcomes for each shadow (qubit-wise) are projected into a higher-dimensional embedding space ($d_{model}=64$).
- **Multi-Head Attention**: The model uses 4 attention heads to weigh the importance of different shadows when estimating the global state.
- **Transformer Layers**: 2 layers of transformer encoding are used to extract deep features from the shadow sequence.
- **Global Pooling**: The output sequence from the transformer is average-pooled over the shadow dimension to obtain a single state representation.

## 3. Enforcing Physical Constraints
The pooled features are fed into dense layers that predict the parameters of a **Lower Triangular Matrix (Cholesky Factor)** $L$.
The final density matrix estimate $\hat{\rho}$ is constructed as:
$\hat{\rho} = \frac{LL^\dagger}{\text{Tr}(LL^\dagger)}$

This mathematical structure guarantees:
1. **Hermiticity**: $LL^\dagger = (LL^\dagger)^\dagger$.
2. **Positive Semi-Definiteness**: $LL^\dagger \ge 0$.
3. **Unit Trace**: Normalization by the trace ensures $\text{Tr}(\hat{\rho}) = 1$.

## 4. Training Objective
The model is trained to minimize the **Frobenius Norm** distance between the predicted $\hat{\rho}$ and the ground-truth density matrix $\rho_{gt}$:
## 5. Conclusions
The refined architecture for Track 1 demonstrates:
- **Robustness**: The self-attention mechanism significantly improves feature extraction from randomized Pauli measurement outcomes compared to traditional MLPs.
- **Physical Compliance**: By hard-coding the Cholesky factor reconstruction, the model's output space is strictly limited to the manifold of valid density matrices, preventing non-physical results.
- **Generalization**: The model generalizes well to unseen Haar random states, maintaining stable performance across different regions of the Hilbert space.
