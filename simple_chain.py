from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#Config
model_file = "models/Vinallama Chat Q5 0.gguf"

#Load LLM
def load_LLM(model_file):
    llm = CTransformers(
        model = model_file,
        model_type ='llama',
        max_new_tokens = 1024,
        temperature = 0.01
    )
    return llm

#Create prompt template
# from langchain.prompts import ChatPromptTemplate
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

#create simple_chain
def create_simple_chain(prompt_t, llm):
    llm_chain = LLMChain(prompt=prompt_t, llm = llm)
    return llm_chain

#from hugging face
template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm =  load_LLM(model_file)
llm_chain = create_simple_chain(prompt, llm)

question = "Chủ tịch nước Việt Nam"
response = llm_chain.invoke({"question": question})
print(response)

