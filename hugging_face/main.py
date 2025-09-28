from transformers import pipeline

# TODO
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype="float16",
    load_in_4bit=True
)

result = pipe("Explain relativity in simple terms:", max_new_tokens=100)
print(result[0]["generated_text"])