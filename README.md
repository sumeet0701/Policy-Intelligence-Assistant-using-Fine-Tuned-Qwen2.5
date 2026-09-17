# Policy Intelligence Assistant
### Fine-Tuning Qwen2.5-1.5B for Enterprise Policy Understanding using QLoRA + RAG

## Overview

Organizations maintain thousands of pages of policy documents covering:

- Promotion Policies
- Leave Policies
- HR Policies
- Travel Policies
- Asset Policies
- Performance Review Policies
- Compliance Policies

Employees often struggle to find precise answers from lengthy documents.

This project demonstrates how to build a Policy Intelligence Assistant by combining:

1. Qwen2.5-1.5B-Instruct
2. QLoRA Fine-Tuning
3. Instruction Dataset Generation
4. Retrieval-Augmented Generation (RAG)

The final solution provides accurate, policy-grounded responses while minimizing hallucinations and reducing infrastructure costs.

---

# Problem Statement

Traditional keyword search suffers from several limitations:

- Poor semantic understanding
- No contextual reasoning
- Difficulty answering scenario-based questions
- Inability to explain policy rules

General-purpose LLMs also struggle because:

- They do not understand organization-specific policies
- They hallucinate missing information
- They lack domain terminology

The objective of this project is to create a domain-specialized Small Language Model (SLM) capable of answering policy-related questions accurately.

---

# Solution Architecture

Policy Documents

       │
       ▼

Document Processing
       │
       ▼

Instruction Dataset Generation
       │
       ▼

QLoRA Fine-Tuning
       │
       ▼

Fine-Tuned Qwen2.5-1.5B
       │
       ▼

Policy Assistant

---

# Why Fine-Tuning?

RAG alone retrieves information.

However, RAG does not teach the model:

- Policy language
- Organization terminology
- Expected response style
- Policy interpretation patterns

Fine-tuning teaches the model:

- Domain vocabulary
- Policy reasoning
- Response structure
- Out-of-scope behavior

---

# Why Not Fine-Tune Everything?

Policies change frequently.

Retraining every time a policy changes is expensive.

Therefore:

Fine-Tuning → teaches behavior

RAG → provides knowledge

The combination provides the best balance between accuracy and maintainability.

---

# Dataset Creation

## Source Documents

The dataset was created from enterprise policy documents.

Examples include:

- Promotion Policy
- Leave Policy
- Compensation Policy
- Travel Policy
- Asset Allocation Policy
- Compliance Guidelines

---

## Dataset Transformation

Each policy section was transformed into instruction-answer pairs.

Example:

### Raw Policy

Eligibility:

Full-time employees who have completed 12 months of service and are not serving notice period are eligible.

### Generated Training Sample

User:

Who is eligible for promotion review?

Assistant:

Employees who have completed 12 months of service and are not serving notice period are eligible.

---

# Training Data Format

The model was trained using OpenAI-style chat messages.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Policy Assistant."
    },
    {
      "role": "user",
      "content": "Who is eligible for promotion review?"
    },
    {
      "role": "assistant",
      "content": "Employees who completed 12 months of service and are not on notice period are eligible."
    }
  ]
}
````

---

# Model Selection

## Base Model

Qwen/Qwen2.5-1.5B-Instruct

Reasons:

* Strong instruction-following capability
* Small deployment footprint
* Efficient fine-tuning
* Suitable for enterprise workloads
* Supports long context windows

Qwen2.5 models are instruction-tuned and support chat-template based supervised fine-tuning. ([Filestore][1])

---

# Fine-Tuning Approach

We use QLoRA (Quantized Low-Rank Adaptation).

Benefits:

* Lower GPU memory usage
* Faster training
* Train only adapter weights
* Preserve base model weights

QLoRA adapts a model through low-rank adapters while keeping most parameters frozen, making small-model fine-tuning practical on modest hardware. ([knowledge-nlp.github.io][2])

---

# Quantization Configuration

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
```

---

# LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none"
)
```

---

# Training Configuration

```python
learning_rate = 2e-4
batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs = 3
warmup_ratio = 0.05
weight_decay = 0.01
```

---

# Training Pipeline

Step 1

Load Dataset

```python
dataset = load_dataset("json")
```

Step 2

Apply Qwen Chat Template

```python
tokenizer.apply_chat_template()
```

Step 3

Tokenization

```python
tokenizer(...)
```

Step 4

Attach LoRA Adapters

```python
get_peft_model(...)
```

Step 5

Supervised Fine-Tuning

```python
Trainer(...)
```

Step 6

Save Adapter

```python
model.save_pretrained("./qwen2.5-finetuned")
```

---

# Inference Architecture

User Question
│
▼
Embedding Model
│
▼
Vector Search
│
▼
Top-K Policy Chunks
│
▼
Prompt Construction
│
▼
Fine-Tuned Qwen2.5
│
▼
Answer

---

# RAG Integration

The fine-tuned model is not used alone.

During inference:

1. User asks a question.
2. Relevant policy chunks are retrieved.
3. Retrieved context is injected into the prompt.
4. Fine-tuned Qwen generates the final answer.

This ensures:

* Current policy information
* Reduced hallucinations
* Traceable answers
* Easy policy updates

---

# Loading Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

model = PeftModel.from_pretrained(
    base_model,
    "./qwen2.5-finetuned"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen2.5-finetuned"
)
```

---

# Project Structure

```bash
.
├── data/
│   └── slm_instruction_finetuning_data.jsonl
│
├── notebooks/
│   └── finetune-qwen2-5-1-5b-model.ipynb
│
├── models/
│   └── qwen2.5-finetuned/
│
├── rag/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── embeddings.py
│   └── prompt_builder.py
│
└── README.md
```

---

# Future Improvements

* Hybrid Search (BM25 + Vector Search)
* GraphRAG
* Policy Version Tracking
* Multi-Document Reasoning
* Citation Generation
* Role-Based Policy Retrieval
* Evaluation using RAGAS

---

# Key Takeaways

* Fine-Tuning teaches behavior.
* RAG provides knowledge.
* QLoRA reduces training cost.
* Small Language Models can perform effectively on enterprise policy tasks.
* Fine-Tuned Qwen2.5 + RAG provides an efficient and scalable policy assistant architecture.

```

For a public GitHub repository, I would also add:
- Training screenshots
- GPU/VRAM requirements
- Dataset statistics (number of samples, avg tokens, max sequence length)
- Training loss curves
- Before vs After fine-tuning examples
- Evaluation metrics (ROUGE, BERTScore, Exact Match, RAGAS) if available

Those sections usually make the repository look significantly more professional and reproducible.
```

[1]: https://filestore.giiisp.com/arxiv/2025/01/06/2412.15115v2.pdf?utm_source=chatgpt.com "2025-01-06
Qwen2.5 Technical Report
Qwen Team
http"
[2]: https://knowledge-nlp.github.io/naacl2025/papers/13.pdf?utm_source=chatgpt.com "3.5 Fine-tuning"
