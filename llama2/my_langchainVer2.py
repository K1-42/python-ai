### NOT WORKED ###
import logging
import sys

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# ドキュメントの読み込み
with open("knowledge_db_hirogaru_en.txt") as f:
    test_all = f.read()

# チャンクの分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # チャンクの最大文字数
    chunk_overlap=20,  # オーバーラップの最大文字数
)
texts = text_splitter.split_text(test_all)

# チャンクの確認
print(len(texts))
for text in texts:
    print(text[:10].replace("\n", "\\n"), ":", len(text))


# インデックスの作成
index = FAISS.from_texts(
    texts=texts,
    embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
)
index.save_local("storage")

# インデックスの読み込み
# index = FAISS.load_local(
#    "storage", HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# )

# LLMの準備
# llm = OpenAI(temperature=0, verbose=True)
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.Q2_K.gguf",
    input={
        "max_tokens": 32,
        "stop": ["System:", "User:", "Assistant:", "\n"],
    },
    verbose=True,
)

# 質問応答チェーンの作成
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.as_retriever(search_kwargs={"k": 4}),
    verbose=True,
)

# 質問応答チェーンの実行
print("A1:", qa_chain.run("What is Hirogaru Sky! PreCure?"))
print("A2:", qa_chain.run("What kind of person is Sora Harewatar?"))
print("A3:", qa_chain.run("Who is Cure Butterfly?"))
