from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from my_library import my_print

################################
# データ登録
# https://qiita.com/rairaii/items/f365d96bb11b72f9ea78
# https://qiita.com/utanesuke/items/6efc03eca94f7de3b9cd
# pip install langchain
# pip install chroma
# pip install CSVLoader
# pip install loader
################################
def insert_data(interface = None):
    my_print('データ登録の開始', interface, add_time = True)

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
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./my_chroma")
    vectorstore.persist()

    # 終了日時
    my_print('データ登録の完了', interface, add_time = True)
