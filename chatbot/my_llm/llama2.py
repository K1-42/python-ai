from llama_cpp import Llama

################################
# LLMに質問
################################
def query(user_message):

    # AIモデルの読み込み
    llm = Llama(model_path="./my_model/llama-2-7b-chat.Q2_K.gguf")

    # 問い合わせ
    llm_return = llm(
        user_message,
        max_tokens=300,
        echo=False,
    )
    
    # AIモデルの回答を抽出
    answer = llm_return['choices'][0]['text'].rstrip('\r\n')
    
    print('========== デバッグ：回答開始 ==========')
    print(str(answer))
    print('========== デバッグ：回答終了 ==========')
    
    return answer

################################
# LLMに質問（RAG）
################################
def query_with_db(user_message):
    
    return '未作成：LLama2のDB版'
