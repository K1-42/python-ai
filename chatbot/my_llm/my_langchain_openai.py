from my_library import my_print

#https://qiita.com/ksonoda/items/ba6d7b913fc744db3d79
from langchain.chat_models import ChatOpenAI
import os
my_print('========= LLMをセットアップ =========', add_time = True)
os.environ["OPENAI_API_KEY"] = 'xxxxxxxxxxxxxxxxx'
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

#### ベクトルDB
from langchain.chains import VectorDBQAWithSourcesChain
my_print('========= ベクトルDBをセットアップ =========', add_time = True)
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
db = Chroma(embedding_function=embeddings, persist_directory="./my_chroma_pdf_openai")
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
