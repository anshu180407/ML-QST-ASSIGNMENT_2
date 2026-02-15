# Replication Guide: Reproducing Neural QST Results

Follow these steps to reproduce the training and evaluation of the 2-qubit Quantum State Tomography model.

## 1. Environment Setup
Install the necessary dependencies using pip:
```bash
pip install pennylane torch scipy tqdm numpy
```

## 2. Dataset Generation
Generate the classical shadows dataset (200 random states with 50 shadows each):
```bash
python Assignment_2/src/generate_data.py
```
This will create a `Assignment_2/data/assignment_2/shadow_dataset_2q.pt` file.

## 3. Training the Model
Run the training script to optimize the transformer network:
```bash
python Assignment_2/src/train.py
```
The script will:
- Train the Transformer for 50 epochs.
- Enforce physical constraints via Cholesky decomposition.
- Save the final model weights to `Assignment_2/outputs/qst_model_2q.pt`.

## 4. Evaluation
To obtain detailed metrics (Fidelity, Trace Distance, Latency), run:
```bash
python Assignment_2/src/eval.py
```

## Final Results (Track 1 Compliance)
- **Mean Fidelity**: 0.8115
- **Trace Distance**: 0.1773
- **Inference Latency**: 4.0702 ms per state
- **Test Set Size**: 40 states
- **Shadows per State (K)**: 50
