#!pip install langchain
#!pip install openai


##### OpenAIのAPIキーを入力

import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


##### LangChainのPDFローダーを使ってPDFファイルからテキストデータを抽出

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/home/datascience/Langchain/sample.pdf")
documents = loader.load_and_split()


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=4000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(docs)


##### テキストデータをDBにロード
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch

# Embeddingに OpenAIのEmbeddingモデルを定義
embeddings = OpenAIEmbeddings()

# ElasticSearchのアクセスポイントとユーザー、パスワードを定義
url = f"http://<user>:<password>@xxx.xxx.xxx.xxx:9200"

# テキストデータをベクトル化し、indexをlangchainとしてelasticsearchにロード
db = ElasticVectorSearch.from_documents(
    docs, embeddings, elasticsearch_url= url, index_name="langchain"
)

##### DBにロードされたかを確認
from elasticsearch import Elasticsearch

indices = Elasticsearch(url).cat.indices(index='*', h='index').splitlines()
for index in indices:
    print(index)

print(Elasticsearch(url).search(index="langchain"))


##### 入力されたLLMのモデルを定義し、ベクトルストアとの連携を定義
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa =VectorDBQAWithSourcesChain.from_chain_type(llm, chain_type="map_reduce", vectorstore=db)



##### LangChainのAgentとToolsを定義します。この定義がまさにLLMとベクトルストアの連携の要
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool

tools = [
    Tool(
      name = "elasticsearch_searcher",
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


##### Agentが最終的な回答を生成を実行
query = "LangChainとは何ですか？"
question = prompt.format(question=query)
agent.run(question)
