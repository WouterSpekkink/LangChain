from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.llms import Petals
import constants
import os

os.environ["HUGGINGFACE_API_KEY"] = constants.APIKEY

question = "What are important authors on the theory of social practices?"

#template = ("Question: {question}\n"
#            "Give a detailed answer\n"
#            "Answer:")



template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template,
                        input_variables = ["question"])

#repo_id = "tiiuae/falcon-7b-instruct"
#repo_id = "stabilityai/FreeWilly2"

#llm = HuggingFaceHub(
#    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1000}
#    )

#llm = Petals(model_name="bigscience/bloomz-petals")
llm = Petals(model_name="meta-llama/Llama-2-70b-chat-hf")

llm_chain = LLMChain(prompt=prompt,
                    llm=llm,
                    )

print(llm_chain.run(question))
