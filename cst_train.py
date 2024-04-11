from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments

from datasets import load_dataset

from trl import DPOTrainer
from peft import LoraConfig
from peft import prepare_model_for_kbit_training
import torch


dataset = load_dataset("vicgalle/configurable-system-prompt-multitask")["train"]
model_name = "teknium/OpenHermes-2.5-Mistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# add special tokens
if model_name == "teknium/OpenHermes-2.5-Mistral-7B":
    tokenizer.add_special_tokens(
        {
            "pad_token": "</s>",
        }
    )


def template_prompt(system, prompt):
    if system is None:
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        if model_name == "abacusai/bigstral-12b-32k":
            messages = [
                {"role": "user", "content": system + "\n" + prompt},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system,
                },
                {"role": "user", "content": prompt},
            ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt


def template_answer(answer):
    messages = [
        {
            "role": "assistant",
            "content": answer,
        },
    ]
    answer = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return answer


# create new columns
dataset = dataset.map(
    lambda x: {
        "prompt": template_prompt(x["system"], x["prompt"])
    },  # change this according to the dataset!!!
)


dataset = dataset.map(
    lambda x: {"chosen": template_answer(x["chosen"])},
)
dataset = dataset.map(
    lambda x: {"rejected": template_answer(x["rejected"])},
)

print(dataset[0])

# LoRA configuration
peft_config = LoraConfig(
    r=8 * 3,
    lora_alpha=16 * 3,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "k_proj",
        "gate_proj",
        "v_proj",
        "up_proj",
        "q_proj",
        "o_proj",
        "down_proj",
    ],
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
)
model.config.use_cache = False


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


output_name = f"checkpoints/exp_configurable_{model_name}"

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    gradient_checkpointing=True,
    output_dir=output_name,
    logging_steps=1,
)

trainer = DPOTrainer(
    model,
    ref_model=None,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(output_name)
