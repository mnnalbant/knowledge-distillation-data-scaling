# Knowledge Distillation via Data Scaling

**How Much Teacher-Generated Data Does a Student Need?**

This project empirically studies **data scaling laws in knowledge distillation** for large language models.
We investigate how the performance of a **small student model** improves as the amount of **teacher-generated instruction data** increases, while keeping the model architecture and training setup fixed.

The core question is:

> *Can increased quantities of high-quality teacher data compensate for limited student model capacity?*

---

## Overview

* **Teacher model:** GPT-2 XL (≈1.5B parameters)
* **Student model:** GPT-2 Small (≈124M parameters)
* **Distillation method:** Supervised Fine-Tuning (SFT) on teacher-generated responses
* **Dataset:** Alpaca instruction prompts
* **Scaling variable:** Number of teacher-generated samples

  * 500, 2k, 5k, 10k, 15k

The teacher generates responses for Alpaca prompts, and **five separate student models** are trained using increasing amounts of this distilled data. All other training parameters are held constant.

---

## Motivation

Large language models are powerful but expensive to deploy. Knowledge distillation aims to:

* Reduce inference latency
* Lower memory and compute requirements
* Enable deployment on consumer or edge hardware

However, **model size alone does not determine performance**. This project explores the trade-off between:

* **Model capacity** (fixed, small student)
* **Data scale** (amount of teacher supervision)

---

## Experimental Setup

### Training Configuration (Fixed Across Runs)

* Epochs: 2
* Learning rate: 5 × 10⁻⁵
* Batch size: 16
* Objective: Supervised Fine-Tuning (instruction → response)

By fixing all hyperparameters, performance differences arise **only** from the amount of teacher-generated data.

---

## Evaluation Metrics

1. **Perplexity (PPL)**
   Measures language modeling quality and fluency.

2. **Instruction-Following Pass Rate**
   Evaluated on a held-out 50-prompt instruction set.

3. **Efficiency Metrics**

   * Tokens per second (generation speed)
   * Model memory footprint (MB)

These metrics capture both **quality** and **deployment trade-offs**.

---

## Key Results

### Perplexity (Language Quality)

* Sharp improvement from **500 → 2,000 samples**
* Gradual saturation beyond **5,000 samples**
* Diminishing returns after **10k–15k**

**Insight:** Most linguistic gains come early; extra data yields marginal improvements once capacity limits are reached.

---

### Instruction Following

* Pass rates remain between **14%–20%**
* Peak performance around **2,000 samples**
* No monotonic improvement with more data

**Interpretation:**
Distillation improves *fluency and style*, but not complex rule-following.
This reflects GPT-2’s lack of native instruction tuning.

---

### Efficiency

* Smallest student (500 samples):

  * ~127 tokens/sec
  * ~475 MB memory footprint
* Extremely fast and deployable on modest hardware

**Trade-off:**
A small drop in instruction performance can be acceptable for **massive efficiency gains**.

---

## Qualitative Findings

* Responses become more coherent and structured as data increases
* Formatting and style improve before factual depth
* Model capacity limits prevent deeper reasoning gains

---

## Conclusions

* **Data scaling helps — but only up to a point**
* For a 124M parameter model, the **sweet spot is ~5,000 samples**
* Distillation transfers *how the teacher speaks* better than *how it reasons*
* High-quality, diverse data matters more than sheer volume

This highlights the **fundamental capacity limits of small models**, even under strong teacher supervision.

---

## Repository Contents

```
.
├── main.ipynb          # Training, evaluation, and plotting
└── README.md           # Project documentation
```

---

## How to Run

```bash
pip install torch transformers datasets matplotlib
```

Open and run:

```bash
jupyter notebook main.ipynb
```

> ⚠️ Training large models requires GPU support.
> The notebook is structured so results can be reproduced or partially evaluated on CPU.

---

## Notes & Limitations

* GPT-2 is **not instruction-tuned** → limits instruction-following gains
* Teacher generation cost is non-trivial
* Results are specific to GPT-2-family models and SFT distillation

---

## References

* Alpaca Instruction Dataset
* GPT-2 (Radford et al.)
