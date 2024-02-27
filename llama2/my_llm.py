from llama_cpp import Llama
from my_library import my_print

my_print('AIモデル読込の開始', add_time = True)
# AIモデルの読み込み
llm = Llama(model_path="./llama-2-7b-chat.Q2_K.gguf")
my_print('AIモデル読込の完了', add_time = True)

################################
# LLMに質問
################################
def llm_query(query):
    my_print('質問の開始', add_time = True)
    output = llm(
        query,
        max_tokens=300,
        echo=True,
    )
    my_print('質問の終了', add_time = True)
    my_print('回答：' + output['choices'][0]['text'])
