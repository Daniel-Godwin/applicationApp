# NVIDIA Nemotron Reasoning Challenge - LoRA Fine-Tuning

This project demonstrates how to fine-tune a large language model (LLM) using **LoRA (Low-Rank Adaptation)** for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).  
The task involves learning binary transformation rules from prompts and predicting valid 8-bit binary outputs.

## 📌 Project Overview
- **Base Model:** [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)  
- **Fine-Tuning Method:** LoRA adapters applied to attention layers (`q_proj`, `v_proj`)  
- **Dataset:** Provided by NVIDIA (train.csv, test.csv)  
- **Goal:** Efficiently adapt a large model to structured reasoning tasks without retraining all parameters.

## ⚙️ Setup
Install required libraries:
```bash
pip install -q transformers accelerate sentencepiece
pip install -q peft dataset

## 📂 Data
- **Train Data:** Contains `id`, `prompt`, and `answer` (binary strings).  
- **Test Data:** Contains `id` and `prompt` only.  
- **Format Conversion:** Prompts are converted into instruction style:
  
  Input binary: 00110100
  Output binary:
  

---

## 🧠 Model & Training
1. Load **Mistral-7B-Instruct** with Hugging Face `transformers`.
2. Apply **LoRA adapters** for parameter-efficient fine-tuning.
3. Freeze base model weights; train only LoRA layers.
4. Training configuration:
   - Batch size: 4  
   - Gradient accumulation: 4  
   - Learning rate: 3e-4  
   - Epochs: 3  
   - Mixed precision (`fp16`)  

---

## 🔮 Inference
- **Single Prediction:**  
  ```python
  predict_bit("00110100")  # returns "00011010"
  ```
- **Batch Prediction:**  
  Runs inference across the test set and saves results to `submission.csv`.

---

## 📊 Output
- Predictions are guaranteed to be **valid 8-bit binary strings**.
- Final submission file:  
  ```
  id,predicted
  00066667,00000010
  000b53cf,00000000
  00189f6a,00100000
  ```

---

## 🚀 Why LoRA?
- **Efficiency:** Fine-tunes only small adapter matrices instead of billions of parameters.  
- **Flexibility:** Multiple tasks can be supported by swapping adapters.  
- **Accessibility:** Makes large model fine-tuning feasible on limited hardware (e.g., NVIDIA T4 GPU on Kaggle).

---

## ✅ Key Features
- End-to-end pipeline: data prep → fine-tuning → inference → submission.  
- Hugging Face + PEFT integration for LoRA.  
- Kaggle-ready workflow with CSV submission.  

---

## 📌 Next Steps
- Experiment with different LoRA configurations (`r`, `alpha`, dropout).  
- Try other reasoning datasets or tasks (text classification, translation).  
- Evaluate accuracy and generalization beyond binary transformations.

---

## 📜 License
This project is for educational and competition purposes.  
Base model usage follows Mistral AI’s license [(huggingface.co in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fhuggingface.co%2Fmistralai%2FMistral-7B-Instruct-v0.1").
```

---

This README gives a **professional, competition-ready overview** of your notebook. Do you want me to also add a **"Quick Start" section with exact Kaggle commands** so someone can run it immediately in a Kaggle notebook?
