# VALOR: Official Implementation for AAAI 2026 Alignment Track

**Paper**: *Value-Aligned Prompt Moderation via Zero-Shot Agentic Rewriting for Safe Image Generation*  
**Conference**: AAAI 2026 (Alignment Track)

This repository contains the **official implementation of VALOR**, a zero-shot, agentic prompt moderation framework for **value-aligned and safe text-to-image generation**.  
VALOR leverages large language models as overseer agents to rewrite unsafe or ambiguous prompts while preserving user intent, enabling safer image generation without model retraining.

---

## 📄 Paper

If you find this work useful, please refer to our paper:

> **Value-Aligned Prompt Moderation via Zero-Shot Agentic Rewriting for Safe Image Generation**  
> Xin Zhao, Xiaojun Chen\*, Bingshan Liu, Zeyao Liu, Zhendong Zhao, Xiaoyan Gu  
> *AAAI 2026, Alignment Track*

---

## 🔍 Abstract

Text-to-image (T2I) models such as Stable Diffusion exhibit strong generative capabilities but are vulnerable to unsafe, adversarial, or value-misaligned prompts. Existing defenses often rely on keyword filtering, post-hoc moderation, or costly fine-tuning, which either fail under semantic jailbreaking or degrade generation quality.

We propose **VALOR (Value-Aligned LLM-Overseen Rewriter)**, a **zero-shot agentic framework** for safe image generation. VALOR integrates:

1. **Multi-granular safety detection** (word-level, semantic-level, and value-level),
2. **Intention disambiguation** for cross-modal misalignment,
3. **LLM-guided prompt rewriting** under value-aligned system instructions,
4. **Optional safety-guided regeneration** for residual unsafe outputs.

Extensive experiments across adversarial, ambiguous, and value-sensitive prompts demonstrate that VALOR can reduce unsafe generations by up to **100%**, while preserving prompt usefulness, creativity, and image quality—without any diffusion model retraining.

---

## ✨ Key Features

- ✅ **Zero-shot & training-free**: no modification or fine-tuning of T2I models
- 🧠 **LLM-overseen rewriting** with role-specific system prompts
- 🔍 **Multi-granular NSFW & value detection**
- 🎯 **Intent-preserving moderation**, not blunt refusal
- 🖼️ Compatible with multiple T2I models (SD v1.4, SDXL, SD v3.5, PixArt-α)
- 🤖 Supports multiple LLM backends (DeepSeek, Qwen, Zephyr, LLaMA)

---

## 🧩 Method Overview

### VALOR Pipeline

VALOR follows a **detect → rewrite → verify** pipeline:

1. **Multi-Granular Detection**
   - **Word-level**: keyword-based NSFW filtering
   - **Semantic-level**: embedding similarity to unsafe references
   - **Value-level**: detection of culturally or ethically inappropriate concept combinations

2. **Intention Judgement**
   - Identifies prompts with negation or constraint semantics that T2I models tend to misinterpret  
   - e.g., *“naked running is forbidden”*

3. **LLM-Guided Rewriting**
   - Unsafe or ambiguous prompts are rewritten using an LLM
   - System prompts differ for **NSFW**, **value violation**, and **intent ambiguity**

4. **Safety-Guided Regeneration (Optional)**
   - If the generated image remains unsafe, a lightweight style suffix (e.g., *illustration style*) is appended to guide regeneration

---

## 🚀 Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
```
### Tested Environment

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA 11.x
- 4 × NVIDIA A100 (40GB) recommended for full evaluation

## ▶️ Running VALOR
All execution commands are provided in ```execute.sh```.
```bash
bash execute.sh
```

## 📁 Repository Structure

```bash
.
├── enhanced_agent.py        # Core VALOR agent implementation
├── execute.sh               # Execution entry (recommended)
├── prompts/                 # Prompt datasets (I2P, custom, etc.)
├── outputs/                 # Generated images and logs
├── requirements.txt         # Python dependencies
└── README.md
```



### 📚 Citation
If you use this repository or find our work helpful, please cite our paper:
```bibtex
@article{Zhao2025ValueAlignedPM,
  title={Value-Aligned Prompt Moderation via Zero-Shot Agentic Rewriting for Safe Image Generation},
  author={Xin Zhao and Xiaojun Chen and Bingshan Liu and Zeyao Liu and Zhendong Zhao and Xiaoyan Gu},
  journal={ArXiv},
  year={2025},
  volume={abs/2511.11693},
  url={https://api.semanticscholar.org/CorpusID:283071671}
}
```
