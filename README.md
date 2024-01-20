# llm-notes

| Description | Notebook |
| ----------- | -------- |
| Full finetuning a very small model `EleutherAI/pythia-70m` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/finetune_full_pythia.ipynb) |
| QLoRA finetuning `Llama2-7b` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/finetune_qlora_llama2.ipynb) |
| QLoRA finetuning `Mistral-7b` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/finetune_qlora_mistral.ipynb) |
| LoRA finetuning with DPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/finetune_dpo.ipynb) |
| GGUF quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/quantize_model_with_gguf.ipynb) |
| Merging LLMs with Mergekit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/merging_with_mergekit.ipynb) |


## Finetuning methods

| Finetuning method | Remarks |
| ----------------- | ------- |
| Full Fine-Tuning | Typically requires 4 GPUs with 24GiB of GPU VRAM on a single node multi-GPU cluster and fine-tuning Deepspeed |
| Parameter Efficient Fine-Tuning (PEFT), e.g. LoRA | Typically requires 4 GPUs with 24GiB of GPU VRAM on a single node multi-GPU cluster and fine-tuning Deepspeed |
| Quantization-Based Fine-Tuning (QLoRA)| Could use a single GPU with 16GiB of GPU VRAM for 7b model |


## Quantization methods

### GGUF quantization with llama.cpp

References: [llama.cpp](https://github.com/ggerganov/llama.cpp)

The names of the quantization methods follow the naming convention: "q" + the number of bits + the variant used.

| Quantization method | Remarks |
| ------------------- | ------- |
| `q2_k` | Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors |
| `q3_k_l` | Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K |
| `q3_k_m` | Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K |
| `q3_k_s` | Uses Q3_K for all tensors |
| `q4_0` | Original quant method, 4-bit |
| `q4_1` | Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models |
| `q4_k_m` | Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K **(recommended)** |
| `q4_k_s` | Uses Q4_K for all tensors |
| `q5_0` | Higher accuracy, higher resource usage and slower inference |
| `q5_1` | Even higher accuracy, resource usage and slower inference |
| `q5_k_m` | Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K **(recommended)** |
| `q5_k_s` | Uses Q5_K for all tensors |
| `q6_k` | Uses Q8_K for all tensors |
| `q8_0` | Almost indistinguishable from float16. High resource use and slow. Not recommended for most users |


## Merging LLMs

### Merging models with mergekit

| Merging method | Remarks |
| -------------- | ------- |
| Spherical Linear Interpolation (SLERP) | Only two models each time |
| TIES | Can merge multiple models at a time |
| DARE | Similar to TIES |
| Passthrough | Experimental. Merge LLMs by concatenating layers from different models |
