# Configurable Safety Tuning of LLMs 🛞


## High-level overview

![CST vs DPO](images/cst_example.png)

> CST versus DPO: after fine-tuning with both strategies, the DPO baseline is overly conservative and fails to generate uncensored outputs, even when the system prompt asks so. The CST-tuned model, on the other hand, is able to be controlled at inference-time depending on the system prompt.

![CST](images/CST.png)

> CST is an extension of DPO which leverages opposite system prompts at fine-tuning time while not requiring additional data, just changing the sign of the preference pairs. This allows the model to be controlled at inference time by selecting the system prompt.



## Dataset 

We released the synthetic dataset for the multi-task experiments from the paper. ...


## Configurable Models 

The CST-tuned models are available in the HuggingFace Hub:


