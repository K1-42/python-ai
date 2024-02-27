import datetime

################################
# 文字列出力
################################
def my_print(message, interface = None, clear = False, add_time = False):
    output_message = message
    if clear == True and interface != None:
        interface.delete(1.0, tk.END)

    if add_time == True:
        current_datetime = datetime.datetime.now()
        #output_message = current_datetime.strftime("%Y年%m月%d日 %H時%M分%S秒") + "：" + output_message
        output_message = current_datetime.strftime("%Y/%m/%d %H:%M:%S") + "：" + output_message

    if interface != None:
        interface.insert(tk.END, output_message)
    else:
        print(output_message)
