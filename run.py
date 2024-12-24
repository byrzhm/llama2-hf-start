import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Select GPU with maximum available memory
def get_gpu_with_max_memory():
    max_memory = 0
    best_gpu = 0
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.get_device_properties(i).total_memory
        print(f"GPU {i} memory: {memory}")
        if memory > max_memory:
            max_memory = memory
            best_gpu = i
    return best_gpu

the_best_gpu = get_gpu_with_max_memory()
torch.cuda.set_device(the_best_gpu)

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

# Load model
model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="cuda",
                torch_dtype=torch.float16
        )

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

## Prompt template
prompt_template = """\
[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]
"""

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

## Generate response
print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template.format(prompt=prompt))[0]['generated_text'])
