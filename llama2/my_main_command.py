from my_library import my_print

#my_print('TEST', add_time = True)

#import my_db_insert
#my_db_insert.insert_data_csv()
#my_db_insert.insert_data_txt()

#import my_db_select
#my_db_select.select_data_csv("PreCure")
#my_db_select.select_data_txt("PreCure")

#import my_llm
#my_llm.llm_query("What is Katom?If you don't now this, you say 'no answer'.")
#my_llm.llm_query("What is 'Hirogaru Sky! PreCure'?If you don't now this, you say 'no answer'.")

import my_langchain
#my_langchain.langchain_query("What is langChain?")
#my_langchain.langchain_query("What is 'Hirogaru Sky! PreCure' ?")
my_langchain.langchain_query("When 'Hirogaru Sky! PreCure' broadcasted ?")
my_langchain.langchain_query("Who is 'Sora Harewatar' ?")
my_langchain.langchain_query("What kind of person is 'Sora Harewatar' ?")

#from my_custom_llm import MyCustomLLM
#llm = MyCustomLLM(n=10)
#llm.invoke("This is a foobar thing")

#import my_langchainVer2
