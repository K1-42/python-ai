from llama_cpp import Llama
import tkinter as tk
import datetime

# データ登録
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings

################################
# AIモデルの読込
################################
def read_model():
    # 開始日時
    start_datetime = datetime.datetime.now()
    start_datetime_formatted = start_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, start_datetime_formatted + "：AIモデル読込中")

    # AIモデルの読み込み
    global llm
    llm = Llama(model_path="./llama-2-7b-chat.Q2_K.gguf")
    # 終了日時
    end_datetime = datetime.datetime.now()
    end_datetime_formatted = end_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    output_text.insert(tk.END, "\n" + end_datetime_formatted + "：AIモデル読込が完了しました。質問をどうぞ！")

################################
# データ登録
# https://qiita.com/rairaii/items/f365d96bb11b72f9ea78
# pip install langchain
# pip install chroma
################################
def insert_data():
    # 開始日時
    start_datetime = datetime.datetime.now()
    start_datetime_formatted = start_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, start_datetime_formatted + "：データ登録中")

    loader = CSVLoader(file_path='./knowledge_db.csv',  source_column='source')
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
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma")
    vectorstore.persist()

    # 終了日時
    end_datetime = datetime.datetime.now()
    end_datetime_formatted = end_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    output_text.insert(tk.END, "\n" + end_datetime_formatted + "：データ登録が完了しました。")

################################
# データ読込
# https://qiita.com/rairaii/items/f365d96bb11b72f9ea78
# pip install langchain
# pip install chromadb
################################
def select_data():
    # 開始日時
    start_datetime = datetime.datetime.now()
    start_datetime_formatted = start_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, start_datetime_formatted + "：データ読込中")
    
    # 読込
    db = Chroma(persist_directory="./chroma")
    output_text.insert(tk.END, "\n" + "count = " + str(db._collection.count()))

#    docs = db.similarity_search("chiba")  # top_kはデフォルト4
#    output_text.insert(tk.END, "\n" + docs[0].page_content)
#    output_text.insert(tk.END, "\n" + docs[1].page_content)

#    docs = db.similarity_search_with_score("saitama")
#    output_text.insert(tk.END, "\n" + docs[0][0].page_content)
#    output_text.insert(tk.END, "\n" + docs[1][0].page_content)
#    output_text.insert(tk.END, "\n" + docs[1][0].page_content)
#    output_text.insert(tk.END, "\n" + docs[1][1].page_content)

    # 終了日時
    end_datetime = datetime.datetime.now()
    end_datetime_formatted = end_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    output_text.insert(tk.END, "\n" + end_datetime_formatted + "：データ読込が完了しました。")

################################
# 質問内容の送付
################################
def question():
    # 開始日時
    start_datetime = datetime.datetime.now()
    start_datetime_formatted = start_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    print(start_datetime_formatted + "：質問開始");

    #prompt = "What are your favorite places to visit in Japan for a summer trip?"
    prompt = input_text.get(1.0, tk.END)
    output = llm(
        prompt,
        max_tokens=300,
        echo=True,
    )
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, output['choices'][0]['text'])

    # 終了日時
    end_datetime = datetime.datetime.now()
    end_datetime_formatted = end_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒")
    print(end_datetime_formatted + "：質問終了");

    # 例：「日本の旅行のおすすめスポットは？」：2分18秒
    # CPU：AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx 2.30 GHz
    # メモリ:16MB

################################
# テキストボックスのリサイズ
################################
def on_window_resize(event):
    new_width = event.width
    input_text.config(width=int(new_width / 10))
    output_text.config(width=int(new_width / 10))

################################
# ウィンドウの作成
################################
root = tk.Tk()
root.title("テスト用AI質問画面")

# ウィンドウのサイズを設定
window_width = 800
window_height = 400
root.geometry(f"{window_width}x{window_height}")

# ウィンドウサイズ変更時のイベントを設定
root.bind("<Configure>", on_window_resize)

# ラベルの作成
label1 = tk.Label(root, text="質問を入力してください")
label1.pack()

# 入力用のテキストボックスの作成
input_text = tk.Text(root, height=10, width=int(window_width / 10))
input_text.pack(fill=tk.X)

# ラベルの作成
label2 = tk.Label(root, text="回答が出力されます")
label2.pack()

# 出力用のテキストボックスの作成
output_text = tk.Text(root, height=10, width=int(window_width / 10))
output_text.pack(fill=tk.X)
output_text.delete(1.0, tk.END)
output_text.insert(tk.END, "AIモデルを読み込んでください（10分ほど時間がかかります）")

# データ読込ボタンの作成
button_insert = tk.Button(root, text="データ登録", command=insert_data)
button_insert.pack(side = 'left')

# データ読込ボタンの作成
button_select = tk.Button(root, text="データ読込", command=select_data)
button_select.pack(side = 'left')

# AIモデル読込ボタンの作成
button_model = tk.Button(root, text="AIモデル読込", command=read_model)
button_model.pack()

# 質問送付ボタンの作成
button_question = tk.Button(root, text="質問内容の送付", command=question)
button_question.pack()

# メインループの開始
root.mainloop()
