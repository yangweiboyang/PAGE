import pickle

# 加载pkl文件
with open('dailydialog_DD.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印数据
print(data[8])
