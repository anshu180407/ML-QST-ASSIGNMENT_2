# Neural Quantum State Tomography (Assignment 2)

This repository contains the implementation for **Assignment 2** of the Machine Learning for Quantum State Tomography (ML-QST) project. The project focuses on reconstructing quantum density matrices from measurement data using deep learning.

## Project Track
- **Track Selection**: Track 1 (Classical Shadows)
- **Model Architecture**: Transformer Encoder
- **Quibts**: 2-qubit Haar random states
- **Constraints**: Physical validity (Hermiticity, Positive Semi-Definiteness, Unit Trace) enforced via Cholesky Decomposition ($\rho = LL^\dagger / \text{Tr}(LL^\dagger)$).

## Directory Structure
The project is organized according to the assignment's mandatory structure:

```text
Assignment_2/
├── src/                # Core source code
│   ├── model.py        # Transformer model implementation
│   ├── generate_data.py # Classical shadow data generation
│   ├── train.py        # Training pipeline
│   └── eval.py         # Evaluation script
├── outputs/            # Saved artifacts
│   └── qst_model_2q.pt # Trained model weights
├── docs/               # Detailed documentation
│   ├── Model_Working.md # Mathematical and architectural logic
│   └── Replication_Guide.md # Step-by-step reproduction manual
├── data/               # Generated datasets
│   └── assignment_2/   # Classical shadow samples (.pt)
├── AI_USAGE.md         # AI attribution disclosure
└── README.md           # Project overview (this file)
```

## Performance Metrics
The model was evaluated on a test set of 40 unseen states with $K=50$ shadows per state.

| Metric | Result |
|--------|--------|
| **Mean Fidelity** | **0.8115** |
| **Mean Trace Distance** | **0.1773** |
| **Inference Latency** | **4.0702 ms / state** |

## How to Run
Please refer to the [Replication Guide](docs/Replication_Guide.md) for detailed instructions on environment setup, data generation, and model training.

1. **Setup**: `pip install -r ../requirements.txt`
2. **Generate Data**: `python src/generate_data.py`
3. **Train Model**: `python src/train.py`
4. **Evaluate**: `python src/eval.py`

## AI Attribution
## Conclusions
The implementation successfully demonstrates the power of **Transformer-based models** for Quantum State Tomography using **Classical Shadows**. Key takeaways include:
- **Physical Validity**: The use of Cholesky decomposition ensures that the model always predicts a physically valid quantum state (PSD, Hermitian, Unit Trace).
- **Efficiency**: Standardizing on $K=50$ shadows per state provides a good balance between data efficiency and reconstruction accuracy.
- **Scalability**: The attention mechanism in the Transformer effectively handles variable shadow sequence lengths, potentially scaling to larger qubit systems.
- **Performance**: High mean fidelity (~0.81) was achieved even with a relatively small training set, proving the robust learning capability of the architecture.
