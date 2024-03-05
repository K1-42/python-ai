from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from my_library import my_print
# テスト用一時配置
from langchain.embeddings.openai import OpenAIEmbeddings
import os

################################
# データ登録
# https://qiita.com/rairaii/items/f365d96bb11b72f9ea78
# https://qiita.com/utanesuke/items/6efc03eca94f7de3b9cd
# pip install langchain
# pip install chroma
# pip install CSVLoader
# pip install loader
################################
def insert_data_csv(interface = None):
    my_print('CSVデータ登録の開始', interface, add_time = True)

    loader = CSVLoader(file_path='./knowledge_db.csv',
                       source_column='source',
                       metadata_columns = ['id', 'author'],
                       encoding='utf-8')
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(
        separator = " ",  
        chunk_size = 1000,  
        chunk_overlap  = 0, 
    )
    
    docs = text_splitter.split_documents(data)
    
    # Indicate model for embedding
    embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    
    # Stores information about the split text in a vector store
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./my_chroma_csv")
    vectorstore.persist()

    # 終了日時
    my_print('CSVデータ登録の完了', interface, add_time = True)

def insert_data_txt(interface = None):
    my_print('TXTデータ登録の開始', interface, add_time = True)

    loader = DirectoryLoader(
        "./", 
        glob="knowledge_db_hirogaru_en.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'autodetect_encoding': True}
    )
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator = " ",  
        chunk_size = 1000,  
        chunk_overlap  = 0, 
        length_function=len,
    )
    
    docs = text_splitter.split_documents(data)
    
    # Indicate model for embedding
    embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    
    # Stores information about the split text in a vector store
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./my_chroma_txt")
    vectorstore.persist()

    # 終了日時
    my_print('TXTデータ登録の完了', interface, add_time = True)


#pip install -U langchain-community
#pip install -U langchain-text-splitters
#pip install pypdf
def insert_data_pdf(interface = None):
    my_print('PDFデータ登録の開始', interface, add_time = True)

    loader = DirectoryLoader(
        "./", 
        glob="knowledge_db_hirogaru_en.pdf", 
        loader_cls=PyPDFLoader, 
    )
    data = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        separator = " ",  
        chunk_size = 1000,  
        chunk_overlap  = 0, 
        length_function=len,
    )
    
    docs = text_splitter.split_documents(data)
    
    # Indicate model for embedding
    embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    
    # Stores information about the split text in a vector store
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./my_chroma_pdf")
    vectorstore.persist()

    # 終了日時
    my_print('PDFデータ登録の完了', interface, add_time = True)

#pip install openai
#pip install tiktoken
def insert_data_pdf_openai(api_key, interface = None):
    my_print('PDFデータ登録（OpenAI）の開始', interface, add_time = True)

    loader = DirectoryLoader(
        "./", 
        glob="knowledge_db_hirogaru_en.pdf", 
        loader_cls=PyPDFLoader, 
    )
    data = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        separator = " ",  
        chunk_size = 1000,  
        chunk_overlap  = 0, 
        length_function=len,
    )
    
    docs = text_splitter.split_documents(data)
    
    # Indicate model for embedding
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings()
    
    # Stores information about the split text in a vector store
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./my_chroma_pdf_openai")
    vectorstore.persist()

    # 終了日時
    my_print('PDFデータ登録（OpenAI）の完了', interface, add_time = True)

#pip install unstructured
#pip install pandas
#pip install openpyxl
def insert_data_xls(interface = None):
    my_print('XLSデータ登録の開始', interface, add_time = True)

    loader = DirectoryLoader(
        "./", 
        glob="knowledge_db_hirogaru_en.xlsx", 
        loader_cls=UnstructuredExcelLoader, 
        loader_kwargs={'autodetect_encoding': True}
    )
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator = " ",  
        chunk_size = 1000,  
        chunk_overlap  = 0, 
        length_function=len,
    )
    
    docs = text_splitter.split_documents(data)
    
    # Indicate model for embedding
    embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    
    # Stores information about the split text in a vector store
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./my_chroma_xls")
    vectorstore.persist()

    # 終了日時
    my_print('XLSデータ登録の完了', interface, add_time = True)
