import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# 環境変数の読み込み
from dotenv import load_dotenv

################################
# ChatGPTに質問
################################
def query(user_message):

    load_dotenv('./.env')
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY == '':
        return 'キーが未設定です：ChatGPT'

    #プロンプトテンプレートを作成
    template = """
    あなたは聞かれた質問に答える優秀なアシスタントです。
    """
    # 会話のテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    # AIモデルの読み込み
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=llm)
        
    # 問い合わせ
    answer = conversation.predict(input=user_message)
    
    return answer

################################
# ChatGPTに質問（RAG）
################################
def query_with_db(user_message):
    
    return '未作成です：ChatGPTのDB版'
