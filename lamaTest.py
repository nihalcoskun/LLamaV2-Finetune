import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

output_dir = "/workspace/notebooks/Nihal/outputs_trdataset/checkpoint-1835"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(output_dir)


from datasets import load_dataset
# dataset = load_dataset("text", data_dir="/workspace/notebooks/Nihal/trwiki-67")
# test_dataset = dataset['train']
dataset_name = "afkfatih/turkishdataset"
dataset_train = load_dataset( dataset_name , split='train[:80%]')
dataset_val = load_dataset( dataset_name , split='train[80%:90%]')
dataset_test = load_dataset( dataset_name , split='train[90%:]')

print(dataset_train)
print(dataset_val)
print(dataset_test)


metin = dataset_test[85]['text'].split("###")[1][7:]
prompt = "### Human:"+metin+" ### Assistant: "

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9, temperature=0.9)

print(f"Üretilen Metin:\n\n {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")