from my_library import my_print

#### LangChain
#https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.vector_db.VectorDBQAWithSourcesChain.html
from langchain.chains import VectorDBQAWithSourcesChain

#### LLM
#https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.chat_models
#from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(model_name="gpt-3.5-turbo")

#from langchain.chat_models import ChatOllama
#llm = ChatOllama(model="./llama-2-7b-chat.Q2_K.gguf")

#https://note.com/npaka/n/n3164e8b24539
my_print('========= LLMをセットアップ =========', add_time = True)
from langchain.llms import LlamaCpp
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.Q2_K.gguf",
    input={
        "max_tokens": 32,
        "stop": ["System:", "User:", "Assistant:", "\n"],
    },
    verbose=True,
    n_ctx=2048, # ValueError: Requested tokens (521) exceed context window of 512
)

#### ベクトルDB
my_print('========= ベクトルDBをセットアップ =========', add_time = True)
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
db = Chroma(embedding_function=embeddings, persist_directory="./my_chroma_pdf")

qa =VectorDBQAWithSourcesChain.from_chain_type(llm, chain_type="map_reduce", vectorstore=db)

##### LangChainのAgentとToolsを定義します。この定義がまさにLLMとベクトルストアの連携の要
my_print('========= LangChainをセットアップ =========', add_time = True)
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

#template = """
#下記の質問に日本語で答えてください。
#質問：{question}
#回答：
#"""

template = """
Please answer the questions below.
questions:{question}
answer：
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)


################################
# 質問
################################
def langchain_query(query):
    my_print('質問を開始しました', add_time = True)
    question = prompt.format(question=query)
    agent.run(question)
    my_print('回答が完了しました', add_time = True)
