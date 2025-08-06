# 🧠 Fine‑Tune LLAMA 2 Using LoRA & QLoRA Techniques

## Introduction

This repository demonstrates efficient fine‑tuning of **Meta’s LLaMA 2** models using **LoRA** (Low‑Rank Adaptation) and **QLoRA** (Quantized LoRA) to adapt large language models using significantly less memory and compute. It’s ideal for tune‑ups on custom datasets or domain‑specific tasks, even on a single GPU.

---

## 📌 Table of Contents

- [Introduction](#introduction)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Dataset Preparation](#dataset-preparation)  
- [LoRA Training Pipeline](#lora-training-pipeline)  
- [QLoRA Training Pipeline](#qlora-training-pipeline)  
- [Inference](#inference)  
- [Configuration & Hyperparameters](#configuration--hyperparameters)  
- [Examples](#examples)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)

---

## Features

- ✅ Supports **LLaMA 2** fine‑tuning with **LoRA** and **QLoRA**  
- ✅ Configurable training via code or YAML config files  
- ✅ Custom dataset support (e.g. interactive instruction‑response format)  
- ✅ Efficient workflows: 4‑bit quantization (NF4), double quantization, paged optimizers  
- ✅ Training on limited GPU memory resources  
- ✅ Save, merge, and export fine‑tuned weights for inference use

---

## Requirements

- Python 3.8+  
- CUDA‑enabled GPU  
- Hugging Face Transformers, PEFT, BitsAndBytes, Accelerate, TRL, Datasets  
- Access to **LLaMA 2 model weights** via Hugging Face (requires Meta license acceptance)

---

## Installation

```bash
git clone <your‑repo‑URL>
cd "Fine Tune LLAMA 2 Using LoRA & QLoRA Techniques"

# Create a virtual environment
python –m venv venv && source venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset Preparation

1. Prepare a dataset with prompt–response pairs (e.g., instruction, input context, response).  
2. Format each example into the pattern:
   ```
   Below is an instruction that describes a task. Write a response that appropriately completes the request.

   ### Instruction:
   <instruction text>

   Input:
   <context if exists>

   ### Response:
   <response>

   ### End
   ```
3. Tokenize and truncate using a tokenizer (e.g. via Hugging Face `AutoTokenizer`) to maximum length.  
4. Remove unnecessary fields and optionally shuffle and split into train/validation sets.

---

## LoRA Training Pipeline

1. Load LLaMA 2 base model (e.g. `meta-llama/Llama-2-7b-hf`) and tokenizer via Hugging Face API.  
2. Freeze base parameters; apply **LoRA adapters** to target modules (e.g. Q, K, V projections with rank `r=16`, `alpha=64`) using PEFT.  
3. Enable gradient checkpointing and prepare for k-bit training.  
4. Print total vs trainable_params to verify LoRA-only fine‑tuning (usually < 1% trainable).  
5. Set up `TrainingArguments` and launch Hugging Face `Trainer` with LoRA‑wrapped model.  
6. Save final checkpoint periodically.  

---

## QLoRA Training Pipeline

1. Use `BitsAndBytesConfig` to enable 4-bit mode (`load_in_4bit=true`), choose quantization type (`nf4`) and double quantization.  
2. Load the model in quantized state and wrap with LoRA adapters.  
3. Train similarly using `paged_adamw_8bit`, gradient accumulation, and fp16 or bf16 compute.  
4. Only LoRA adapters are updated during training; the base model stays frozen.  
5. Save LoRA checkpoint and then **merge** weights into final merged directory using `.merge_and_unload()` on `AutoPeftModelForCausalLM`.  
6. Save merged model and tokenizer for easy inference.

---

## Inference

After merging LoRA weights:

```python
from transformers import AutoTokenizer
from peft import PeftModel

model = PeftModel.from_pretrained("results/llama2/final_merged_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("results/llama2/final_merged_checkpoint")
tokenizer.pad_token = tokenizer.eos_token

# Simple generation example
input_prompt = "Your prompt here."
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Configuration & Hyperparameters

| Parameter             | Recommended |
|-----------------------|-------------|
| LoRA rank (r)         | 8 or 16     |
| LoRA alpha            | 16 × r      |
| Modules targeted      | `q_proj`, `v_proj` (optionally `k_proj`, `output_proj`) |
| LoRA dropout          | 0.1         |
| Batch size            | 1 or 2 per device |
| Gradient accumulation | 2–4 steps   |
| Optimizer             | `paged_adamw_8bit` |
| fp16 / bf16           | Usually `fp16` (CUDA) or `bfloat16` (ROCm) |
| Max training steps or epochs | Adjustable (e.g. 20 steps or several epochs) |
| Quantization type     | NF4 (recommended) |
| Double quantization   | Enabled |
| Validation split / early stopping | Optional, via HF `EarlyStoppingCallback` |

---

## Examples

- **LoRA-only run**: Fine-tune with base model frozen, small dataset.  
- **QLoRA run**: Fine-tune with base model quantized to 4‑bit using NF4, LoRA adapters applied, merge results for inference.

Sample notebook or script is included: `Fine_tune_Llama_2.ipynb`.

---

## Troubleshooting

- **No GPU detected?** Ensure correct CUDA or ROCm environment and `torch.cuda.is_available()` returns `True`.  
- **OOM errors**: Reduce LoRA rank, batch size, or enable gradient accumulation.  
- **Merged model performance is degraded**: verify correct `.merge_and_unload()` workflow and consistent quantization configuration.  
- **Incorrect tokenization or prompt format**: Check prompt formatting and ensure EOS/pad tokens are handled correctly.

---

## Contributing

Contributions welcome! Please open issues or pull requests for suggestions, support of additional datasets, improved pipeline scripts, or real-world benchmarks.

---

## License

MIT License — see the [LICENSE](LICENSE) file.

---

## 🔬 References

- QLoRA: efficient fine‑tuning of quantized LLMs (4-bit quantization, NF4, paged optimizer)  
- LoRA methodology (low‑rank adapters, trainable parameters only)  
- OVHcloud and ROCm guides on single‑GPU fine‑tuning processes  
