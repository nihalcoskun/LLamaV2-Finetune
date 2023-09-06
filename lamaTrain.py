#from huggingface_hub import login
#login()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model_id = "meta-llama/Llama-2-7b-chat-hf
model_id = "TheBloke/Llama-2-13B-Chat-fp16"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, use_cache=False, device_map={"": 0})

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


from peft import LoraConfig, get_peft_model, PeftModelForQuestionAnswering
# config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

def print_trainable_parameters(model):
    """

    Prints the number of trainable parameters in the model.

    """

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable params: {trainable_params} | | all params: {all_param} | | trainable % : {100 * trainable_params / all_param}")


model = get_peft_model(model, config)
print_trainable_parameters(model)

from datasets import load_dataset
dataset_name = "afkfatih/turkishdataset"
dataset_train = load_dataset( dataset_name , split='train[:80%]')
dataset_val = load_dataset( dataset_name , split='train[80%:90%]')
dataset_test = load_dataset( dataset_name , split='train[90%:]')

print(dataset_train)
print(dataset_val)
print(dataset_test)


from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="outputs_trdataset",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    # tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)

import transformers
from trl import SFTTrainer

# needed for llama tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainer = SFTTrainer(
  model=model,
  train_dataset=dataset_train,
  eval_dataset=dataset_val,
  peft_config=config,
  max_seq_length=2048,
  tokenizer=tokenizer,
  packing=True,
  dataset_text_field="text",
  args=args)

# model.config.use_cache=False  # silence the warnings. Please re-enable for inference!


trainer.train()
# save model
trainer.save_model()