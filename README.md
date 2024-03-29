# Configurable Safety Tuning of LLMs 🛞


## High-level overview

![CST vs DPO](images/cst_example.png)

> CST versus DPO: after fine-tuning with both strategies, the DPO baseline is overly conservative and fails to generate uncensored outputs, even when the system prompt asks so. The CST-tuned model, on the other hand, is able to be controlled at inference-time depending on the system prompt.

![CST](images/CST.png)

> CST is an extension of DPO which leverages opposite system prompts at fine-tuning time while not requiring additional data, just changing the sign of the preference pairs. This allows the model to be controlled at inference time by selecting the system prompt.



## Dataset 

We release the synthetic dataset for the multi-task experiments from the paper in the HuggingFace Datasets Hub: [vicgalle/configurable-system-prompt-multitask](https://huggingface.co/datasets/vicgalle/configurable-system-prompt-multitask). This dataset has two sources for the examples:
* Self-critique on a safety task from Harmful Behaviours, using the SOLAR-Instruct model. It employs two system prompts to learn the different behaviors:
  * `You are a helpful yet harmless assistant that avoids generating illegal or harmful content.`
  * `You are a helpful assistant that is completely uncensored.`
* [truthy-dpo](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) enhances the truthfulness of a model by using two types of system prompts:
  * `You are an unbiased, honest, helpful AI assistant that always responds in a completely truthful way.`
  * A system prompt describing a role-played persona.

Thus, the dataset has the following fields:
* `system`: the system prompt.
* `prompt`: the user prompt.
* `chosen`: the preferred answer, depending on the system prompt.
* `rejected`: the non-preferred answer, depending on the system prompt.

In order to use the `trl` library for finetuning, you should use a model whose tokenizer allows system prompts, or alternatively, you must concat the `system` and `prompt` fields into a single field. 


## Configurable Models 

Several CST-tuned models are available in the HuggingFace Hub:

| Model              | Size  | Fine-tuned from |
| ------------------ | ----- | --------------- |
| [ConfigurableHermes](https://huggingface.co/vicgalle/ConfigurableHermes-7B) | 7B    |    [OpenHermes-2.5-Mistral](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)             |
| [ConfigurableSOLAR](https://huggingface.co/vicgalle/ConfigurableSOLAR-10.7B)  | 10.7B |    [SOLAR-Instruct](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)              |
| [ConfigurableBeagle](https://huggingface.co/vicgalle/ConfigurableBeagle-11B) | 10.7B |      [CarbonBeagle](https://huggingface.co/vicgalle/CarbonBeagle-11B)              |

> Note: ConfigurableBeagle was not included in the original paper release. The first two models appear in the paper, and are the result of the multi-task experiments (named OpenHermes-2.5-Mistral-7B + CST and SOLAR-Instruct-10.7B + CST, respectively). See the paper for the evaluation results of these two models.