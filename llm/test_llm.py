from llm.tinyllama import TinyLlamaLLM

llm = TinyLlamaLLM()

prompt = "Answer in one sentence: What is retrieval augmented generation?"
response = llm.generate(prompt)

print("LLM response:\n")
print(response)
