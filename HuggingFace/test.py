from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import constants
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = constants.APIKEY

question = "What is a definition of social practices?"

template = ("Question: {question}\n"
            "Answer: Let's thik step by step.")

prompt = PromptTemplate(template=template,
                        input_variables = ["question"])

repo_id = "tiiuae/falcon-7b-instruct"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1000}
    )

llm_chain = LLMChain(prompt=prompt,
                    llm=llm,
                    )

print(llm_chain.run(question))
