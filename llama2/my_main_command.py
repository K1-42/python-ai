from my_library import my_print
from my_custom_llm import MyCustomLLM

#my_print('TEST', add_time = True)

#import my_db_insert
#my_db_insert.insert_data()

#import my_db_select
#my_db_select.select_data("株式会社カトムの設立")

#import my_llm
#my_llm.llm_query("株式会社カトムはどこにありますか？")

###import my_langchain
###my_langchain.langchain_query("株式会社カトムはどこにありますか？")


llm = MyCustomLLM(n=10)
llm.invoke("This is a foobar thing")
