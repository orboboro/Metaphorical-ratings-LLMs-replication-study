from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

model = pipeline("", model="", device="cpu")

llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template(
    "Summarize the following text in a way a {age} year old would understand:\n\n{text}"
)

summarizer_chain = template | llm

text_to_summarize = input("\nEnter text to summarize:\n")
age = input("Enter target age for simplification:\n")

summary = summarizer_chain.invoke({"text": text_to_summarize, "age": age})

print("\n🔹 **Generated Summary:**")
print(summary)
