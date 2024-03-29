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

| Model              | Size  | Fine-tuned from |
| ------------------ | ----- | --------------- |
| [ConfigurableHermes](https://huggingface.co/vicgalle/ConfigurableHermes-7B) | 7B    |    [OpenHermes-2.5-Mistral](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)             |
| [ConfigurableSOLAR](https://huggingface.co/vicgalle/ConfigurableSOLAR-10.7B)  | 10.7B |    [SOLAR-Instruct](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)              |
| [ConfigurableBeagle](https://huggingface.co/vicgalle/ConfigurableBeagle-11B) | 10.7B |      [CarbonBeagle](https://huggingface.co/vicgalle/CarbonBeagle-11B)              |

> Note: ConfigurableBeagle was not included in the original paper release. The first two models appear in the paper, and are the result of the multi-task experiments (named OpenHermes-2.5-Mistral-7B + CST and SOLAR-Instruct-10.7B + CST, respectively). See the paper for the evaluation results of these two models.