# Setup Guide: Atheria on Kaggle / Colab

This guide explains how to set up the Atheria environment on cloud notebook platforms like Kaggle or Google Colab to leverage their GPUs (T4 x2, P100, etc.) for training and simulation.

## 1. Prerequisites

-   A Kaggle account or Google account.
-   Access to a notebook with GPU acceleration enabled.

## 2. Installation Steps

Copy and paste these commands into the first cell of your notebook.

### Step 2.1: Clone Repository
```bash
!git clone https://github.com/Jonakss/Atheria.git
%cd Atheria
```

### Step 2.2: Install Dependencies
```bash
!pip install -r requirements.txt
# Ensure PyTorch is installed with CUDA support (usually pre-installed on Kaggle/Colab)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2.3: Compile Native Engine (C++)
The native engine requires compilation. We use `setup.py` which invokes CMake.

```bash
# Install build tools if missing
!apt-get update && apt-get install -y cmake build-essential

# Compile and install the extension
!python setup.py install
```

### Step 2.4: Verify Installation
Run the test script to ensure everything is working.

```bash
!python scripts/test_native_infinite_universe.py
```

## 3. Running Experiments

### Training
To train a model, use the `train.py` script.

```python
!python src/train.py --experiment my_experiment --config configs/default.yaml
```

### Stress Testing
To benchmark the engine performance.

```bash
!python scripts/stress_test_native.py --engine native --grid_size 1024 --density 0.2 --steps 100
```

## 4. Tips for Kaggle/Colab

-   **Persistence**: Data in `/kaggle/working` (Kaggle) or `/content` (Colab) is lost after the session.
    -   **Kaggle**: Save outputs to a dataset or download them.
    -   **Colab**: Mount Google Drive:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
-   **GPU Info**: Check your GPU with `!nvidia-smi`.
-   **Runtime**: Kaggle sessions can last up to 9-12 hours. Colab varies.

## 5. Troubleshooting

-   **CMake Error**: If CMake fails, check `!cmake --version`. You might need to upgrade it:
    ```bash
    !pip install cmake --upgrade
    ```
-   **CUDA Mismatch**: Ensure the PyTorch CUDA version matches the system CUDA driver. Run `!nvcc --version` to check system CUDA. If needed, reinstall PyTorch with the correct CUDA version:
    ```bash
    !pip uninstall torch -y
    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
