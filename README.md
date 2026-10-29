# HateGuard AI: Multi-Tiered Guardrail & Human-in-the-Loop Moderation for Indonesian Code-Mixed Social Media

Official repository for research artifacts and software systems of **HateGuard AI** by **Group 2 — Department of Informatics, Faculty of Information Technology, Universitas Tarumanagara**:
- **Michael Respati Sanjaya Ho** (535250008)
- **Felix Lin** (535250044)
- **Charlie Jhodya Soe** (535250180)

This documentation explains the environment specifications, dataset access, training and evaluation protocols, inference mechanisms, and the execution of a smoke test from a clean environment. All evaluation numbers reported in the scientific paper refer to a single source of truth at `results/RESULTS_SINGLE_SOURCE.json`.

---

## 1. Repository Structure

```text
HateGuard-Repository/

├── source/          # Training notebooks, Gradio application notebook, core module (hateguard_core.py), & smoke test script
├── config/          # Experiment configuration (config.json), main manifest, addendum manifest, & requirements_frozen.txt
├── results/         # Main evaluation results (RESULTS_SINGLE_SOURCE.json), CSV tables, & figures/ subfolder (300 DPI)
├── paper/           # Scientific paper manuscript formatted for Jurnal RESTI
├── DATA_CARD.md     # Documentation of dataset provenance, partition composition, and ethical limitations
├── Audit_Report.md  # Methodological audit report, cross-device parity verification, & artifact integrity
├── LICENSE          # Software license (MIT License)
└── .gitignore       # Exclusion rules for large model weights and moderator privacy logs
```
---

## 2. Environment Specifications & Installation

Experiments were executed on a Google Colab GPU environment (NVIDIA Tesla T4, CUDA 13.0) with Python 3.13.15, PyTorch 2.11.0, Transformers 5.20.1, Datasets 4.8.5, and scikit-learn 1.6.1. All pseudo-random number generators were locked to `seed = 42` (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`).

To set up a clean environment on a local machine or new server, run the following command in the terminal:

```bash
pip install -r config/requirements_frozen.txt
```
---

## 3. Dataset Access & Provenance

The system utilizes a combination of two public corpus sources:
1. **Indonesian Twitter Hate Speech Dataset** (Ibrohim & Budi, 2019) alongside a normalization dictionary for informal/alay language (`new_kamusalay.csv`, 15,195 entries). SHA-256 values for raw files are logged in `config/experiment_manifest_addendum.json`.
2. **Code-Mixed Indonesian Hate Speech Dataset** (`aiatums/codemixed-id-hate-speech`, *Indonesian-Javanese* and *Indonesian-Sundanese* configurations, CC BY 4.0 license, commit `19a03eba294171be64a5fcb5dcf7f36b6e86d72f`).

Details on cleaning contradictory labels, deduplication, and leak-free partitioning (`GroupShuffleSplit` based on `group_id`) are thoroughly documented in `DATA_CARD.md`.
---

## 4. Access to Frozen Checkpoint Artifacts & SHA-256 Checksum Verification

In accordance with Git repository management best practices, large model weight files (`model.safetensors`, ~440 MB) are not stored directly within Git. Instead, they are hosted on cloud storage and verified using SHA-256 checksums:
- **Cloud Storage Link (`hateguard_artifacts`)**: `https://drive.google.com/drive/folders/1oKxbmulcBvUbaBUtoGvRC2hwZqyku0vQ?usp=drive_link`
- **Base Model**: `indolem/indobertweet-base-uncased` (Apache-2.0)
- **SHA-256 Model Weights (`trusted_model_sha256`)**: `3ec22197cb5248be993782d490cc080efb9f2d0f341ba7884f1b7d99ff31e0df`
- **SHA-256 Core Module (`hateguard_core.py`)**: `e07ad4c7524b1ed3e30fa0fa5a91a9163c40ab4a0547663afc15ef7f861a91b6`
- **SHA-256 Preprocessing Parity (`preprocessing_parity_hash`)**: `0ebe0f02665f2f7b49c5965ceb4ce35f17ba932da93ff528b4e691d6cc5152f8`

---

## 5. Training, Verification, and Evaluation Procedures

Open `source/HateGuard_Training_Final.ipynb` and select the execution mode via the `RUN_MODE` variable in Cell 1:

- `RUN_MODE = "verify_frozen"` *(Default)*: Verifies the checksum authenticity of frozen model weights, then executes the complete statistical evaluation (1,000 grouped bootstrap resamples in FP32 precision), classical baselines (TF-IDF + Logistic Regression), Zero-Shot LLM comparisons (`Qwen/Qwen2.5-1.5B-Instruct` on N = 400 independent groups), guardrail policy & normalization ablation studies, Temperature Scaling calibration, and publication plot generation without overwriting the original model weights.
- `RUN_MODE = "retrain_new_dir"`: Performs fine-tuning from scratch and saves the results to a separate directory (`./hateguard_artifacts_retrain/`).
---

## 6. Running the Inference Application (Gradio HITL) & Smoke Test
### A. Smoke Test from a Clean Environment (Standalone Terminal)
Download the `hateguard_artifacts` folder from the Google Drive link above, then execute the following command from a terminal outside Jupyter Notebook:

```bash
python source/smoke_test_standalone.py ./hateguard_artifacts cpu

```
The script will verify all SHA-256 checksums, test prediction determinism, and check FP32 probability parity across 200 stratified benchmark samples (tolerance limit <= 0.001). Exit code `0` (`SMOKE TEST PASSED`) indicates the artifact is fully valid.

### B. Interactive Web Application (Gradio & SQLite Audit Dashboard)

Run all cells in `source/HateGuard_App_Final.ipynb`. The application enforces a *Hard Fail* principle on startup if artifact hashes mismatch, lazily loads the `Qwen/Qwen2.5-1.5B-Instruct` assistant specifically for the `STRONG` queue, and records every moderator decision into an SQLite database (`moderator_audit.db`).

---

## 7. Operational Log Security & Ethics

Application operational log files (`moderator_audit.db` and `moderator_log.json`) store processed comment texts and are excluded from the public repository via `.gitignore`. This system is designed as a Human-in-the-Loop decision-support system and is not intended to autonomously replace human moderator judgment.