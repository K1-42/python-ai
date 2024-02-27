#### LangChain
#https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.vector_db.VectorDBQAWithSourcesChain.html
from langchain.chains import VectorDBQAWithSourcesChain

#### LLM
#https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.chat_models
#from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(model_name="gpt-3.5-turbo")

#from langchain.chat_models import ChatOllama
#llm = ChatOllama(model="./llama-2-7b-chat.Q2_K.gguf")

#NG:BaseLanguageModel
#https://python-langchain-com.translate.goog/docs/modules/model_io/llms/custom_llm
from llama_cpp import Llama
llm = Llama(model_path="./llama-2-7b-chat.Q2_K.gguf")

#### ベクトルDB
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
db = Chroma(embedding_function=embeddings, persist_directory="./my_chroma")

qa =VectorDBQAWithSourcesChain.from_chain_type(llm, chain_type="map_reduce", vectorstore=db)

##### LangChainのAgentとToolsを定義します。この定義がまさにLLMとベクトルストアの連携の要
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool

tools = [
    Tool(
      name = "my_searcher",
      func=qa,
      description="Langchainの説明"
  )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

##### LangChainのPromptTemplateを定義
from langchain.chains.qa_with_sources.map_reduce_prompt import QUESTION_PROMPT
from langchain import PromptTemplate

template = """
下記の質問に日本語で答えてください。
質問：{question}
回答：
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)


################################
# 質問
################################
def langchain_query(query):
    question = prompt.format(question=query)
    agent.run(question)
