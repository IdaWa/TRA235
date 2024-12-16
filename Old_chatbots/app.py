from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLaMA 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Use the Hugging Face Hub name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# User input
user_input = "Hi, how are you?"
inputs = tokenizer(user_input, return_tensors="pt").to("cuda")  # Use GPU if available

# Generate response
outputs = model.generate(inputs.input_ids, max_length=100, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
