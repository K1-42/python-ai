##########################################################
# メモ
##########################################################

#https://nuco.co.jp/blog/article/YhjPe4Hf
#pip install streamlit streamlit-chat langchain openai python-dotenv
#touch .env
#streamlit run main.py 

##########################################################
# 初期設定、ライブラリインポート
##########################################################

# 画面用ライブラリ
import streamlit as st
from streamlit_chat import message

# LLM利用のための自作モジュール
from my_llm import llama2
from my_llm import chatgpt

# 定数定義
LLM_LLAMA2 = 'llama2'
LLM_CHATGPT = 'chatGPT'

##########################################################
# イベント
##########################################################

#会話の読み込みを行う関数を定義
@st.cache_resource
def load_conversation(llm_name, use_db, user_message):
    if llm_name == LLM_LLAMA2:
        if use_db:
            answer = llama2.query_with_db(user_message)
        else:
            answer = llama2.query(user_message)
    else:
        if use_db:
            answer = chatgpt.query_with_db(user_message)
        else:
            answer = chatgpt.query(user_message)
    return answer

# 質問と回答を保存するための空のリストを作成
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

# 送信ボタンがクリックされた後の処理を行う関数を定義
def on_input_change(llm_name, use_db):
    user_message = st.session_state.user_message

    answer = load_conversation(llm_name, use_db, user_message)

    st.session_state.generated.append(answer)
    st.session_state.past.append(user_message)
    st.session_state.user_message = ""

##########################################################
# 画面定義
##########################################################
# タイトルやキャプション部分のUI
st.title("My AI Test Chatbot")
st.caption("develop mode.")
st.write("質問に答えます。")

# 会話履歴を表示するためのスペースを確保
chat_placeholder = st.empty()

# 会話履歴を表示
with chat_placeholder.container():
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i],is_user=True)
        message(st.session_state.generated[i])

# 質問入力欄と送信ボタンを設置
with st.container():
    user_message = st.text_input("質問を入力する", key="user_message")
    col1, col2 = st.columns(2)
    with col1:
        st.button("送信（llama2）", on_click=on_input_change, args=(LLM_LLAMA2, False,))
        st.button("送信（ChatGPT）", on_click=on_input_change, args=(LLM_CHATGPT, False,))

    with col2:
        st.button("送信（llama2：DB）", on_click=on_input_change, args=(LLM_LLAMA2,True,))
        st.button("送信（ChatGPT：DB）", on_click=on_input_change, args=(LLM_CHATGPT,True,))
