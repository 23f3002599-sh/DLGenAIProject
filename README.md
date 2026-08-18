# Smart MCQ Solver Challenge — Deep Learning & GenAI Approaches

Project for **BSDA2001P — Introduction to Deep Learning and Generative AI**
IIT Madras, BS in Data Science and Applications | Roll No: 23f3002599

Three deep-learning systems built and compared for the Kaggle **Smart MCQ Solver Challenge**: ranking the top-3 most likely correct answers (of 5 options, A–E) for multiple-choice questions spanning philosophy, physics, astronomy, and general science. Submissions are scored with **Mean Average Precision at 3 (MAP@3)**.

## Results

| Model | Type | Trainable Params | Val. MAP@3 | Relative Cost |
|---|---|---|---|---|
| Weighted heuristic (baseline) | Hand-crafted, no learning | 0 | 0.558 | Negligible |
| **Model 1** — BiLSTM + Attention | RNN, trained from scratch | ~9.8M | 0.664 | Low (CPU/GPU, minutes) |
| **Model 2** — Fine-tuned RoBERTa-base | Fine-tuned transformer encoder | ~125M (full fine-tune) | 0.791 | Moderate (single GPU, ~20–30 min) |
| **Model 3** — MiniLM Retrieval + LoRA Qwen2.5-3B | RAG + PEFT generative pipeline | 7.37M / 3.09B (0.24%) | 0.958* | High (GPU + retrieval + generation latency) |

\* See [Known Issues](#known-issues--caveats) — this figure has a validation-leakage caveat. The properly zero-shot (untuned) RAG checkpoint scored **0.709**, which is the more trustworthy generalisation estimate for this pipeline family.

Full methodology, ablations, and analysis are in [`DL_GenAI_Project_Report.docx`](./DL_GenAI_Project_Report.docx).

## Repository Structure

```
.
├── notebook.ipynb              # Full Kaggle notebook (EDA, all 3 models, submissions)
├── DL_GenAI_Project_Report.docx
├── environment.yml             # Conda environment for local reproduction
├── data/                       # (not included — see Dataset section)
│   ├── train.csv
│   └── test.csv
└── submissions/
    ├── submission_scratch_ranker.csv     # Weighted heuristic baseline
    ├── submission_model2_minilm.csv      # Zero-shot MiniLM ranker
    └── submission_model3_minilm_qwen_lora.csv  # Final: MiniLM + Qwen2.5-3B + LoRA
```

## Dataset

- **Source:** Kaggle competition `smart-mcq-solver-challenge`
- **Train:** 2,000 labelled rows — `id`, `prompt`, `A`–`E` (options), `answer`
- **Test:** 500 rows (unlabelled) — same schema minus `answer`
- Download via the [Kaggle competition page](https://www.kaggle.com/competitions/smart-mcq-solver-challenge) or the Kaggle API:
  ```bash
  kaggle competitions download -c smart-mcq-solver-challenge -p data/
  unzip data/smart-mcq-solver-challenge.zip -d data/
  ```

## Models

### Model 1 — BiLSTM + Additive Attention (from scratch)
A Siamese dual-encoder: prompt and each option are independently embedded (GloVe-initialised, 300-dim), passed through a 2-layer bidirectional LSTM (256 hidden units/direction), and pooled with Bahdanau-style additive attention. Matching layer concatenates `[v_p ; v_o ; |v_p − v_o| ; v_p ⊙ v_o]` into an MLP scoring head. No pre-trained transformer weights — establishes a genuine from-scratch deep-learning baseline.

### Model 2 — Fine-Tuned RoBERTa-base
`roberta-base` loaded via `AutoModelForMultipleChoice` and fully fine-tuned (all ~125M parameters) on the 5-way option classification formulation, with FP16 mixed precision, linear warmup + decay schedule, 3 epochs.

### Model 3 — MiniLM Retrieval + LoRA-Fine-Tuned Qwen2.5-3B (RAG)
A two-stage retrieval-augmented generation pipeline:
1. **Retrieval:** every training question is embedded into a 384-dim space with `all-MiniLM-L6-v2`; cosine similarity retrieves the top-5 most similar previously-solved questions for any query.
2. **Generation:** retrieved examples are inserted as in-context demonstrations into a structured prompt fed to `Qwen/Qwen2.5-3B-Instruct`, loaded in **4-bit NF4 quantization** and fine-tuned with **LoRA** (rank 16, alpha 32, dropout 0.05, targeting `q_proj/k_proj/v_proj/o_proj`) via Hugging Face's `SFTTrainer` (TRL).

Three variants were validated in sequence: zero-shot MiniLM similarity only (0.400) → untuned Qwen2.5 + RAG (0.709) → LoRA fine-tuned (0.958*).

## Setup

### Option A — Conda (local / any GPU box)
```bash
conda env create -f environment.yml
conda activate smart-mcq-solver
```

### Option B — Kaggle Notebook (as originally run)
The notebook was developed and run on Kaggle's free **T4 GPU** runtime. Kaggle's base image already ships a recent `transformers`/`torch`/`peft` stack — avoid `--force-reinstall` or hard version pins against it; install only what's missing with `pip install --no-deps` where possible (see notes inside the notebook's Model 3 section).

If your Kaggle session has **2 GPUs visible**, either set `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` before importing `torch`, or set `trainer.args._n_gpu = 1` before calling `.train()` — otherwise `Trainer` will wrap the model in `DataParallel` and clash with `device_map` placement during LoRA fine-tuning.

### Weights & Biases
All three models log training/validation metrics and submission artifacts to W&B. Set your API key as a Kaggle Secret (`WANDB_API_KEY`) or locally via:
```bash
export WANDB_API_KEY=your_key_here
```
Project: `DLGenAIProject` (entity: `23f3002599-indian-institute-of-technology-madras`)

## Running

Open `notebook.ipynb` and run cells top-to-bottom under each `# Model N` section. Each model section is self-contained (loads data, trains/evaluates, and writes its own `submission_*.csv` to `/kaggle/working/` or the local working directory).

## Known Issues & Caveats

- **Model 3 validation leakage:** the 200-row validation sample used to report the final 0.958 MAP@3 was drawn from the same pool the LoRA adapters were fine-tuned on, so the LLM had already seen the correct answers during training. The properly zero-shot **0.709** figure (RAG without fine-tuning) is the trustworthy generalisation estimate. A pre-registered, strictly held-out split is the priority fix — see report Section 6.3.
- **4-bit NF4 quantization** is required to fit LoRA fine-tuning of the 3B-parameter Qwen2.5 model within a T4's 15.6GB VRAM; skipping it will OOM.
- Zero-shot semantic similarity (MiniLM alone) underperforms even the simple hand-crafted heuristic baseline, since near-duplicate distractors carry very similar topical embeddings — retrieval needs a generative reasoning step on top to be useful here.

## Future Work

- Re-run LoRA fine-tuning with a strict pre-registered held-out split.
- Train directly against the MAP@3 ranking objective rather than single-label next-token prediction.
- Weighted ensemble across all three model families.
- Explore hybrid dense+sparse retrieval and alternative backbones (e.g. DeBERTa-v3).

## References

See report Section 7 for full citations (Vaswani et al. 2017; Devlin et al. 2019; Liu et al. 2019 [RoBERTa]; Hochreiter & Schmidhuber 1997 [LSTM]; Bahdanau et al. 2015 [attention]; Reimers & Gurevych 2019 [Sentence-BERT]; Wang et al. 2020 [MiniLM]; Hu et al. 2021 [LoRA]; Dettmers et al. 2023 [QLoRA]; Qwen Team 2024).

## License

Academic coursework submission — IIT Madras BS in Data Science and Applications. Not licensed for redistribution.
