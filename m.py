import torch
import torch.nn as nn
import torch.nn.functional as F


# affine1 = nn.Parameter(torch.Tensor(300,300))#1024
# affine2 = nn.Parameter(torch.Tensor(300,300))#1024
# emo_emb=torch.randn(4, 5,300)
# utter_emb=torch.randn(1, 3, 200)
# utter_emb= nn.ConstantPad1d((0, 300-200), 0)(utter_emb)

# if emo_emb.size(0) < 4:
#     pad_size = (4 - emo_emb.size(0), emo_emb.size(1), emo_emb.size(2))
#     emo_emb= torch.cat([emo_emb, torch.zeros(pad_size)], dim=0)
# elif emo_emb.size(0) > 4:
#     emo_emb = emo_emb[:4, :, :]
# if utter_emb .size(0) < 4:
#     pad_size = (4 - utter_emb .size(0), utter_emb .size(1), utter_emb .size(2))
#     utter_emb = torch.cat([utter_emb , torch.zeros(pad_size)], dim=0)
# elif utter_emb .size(0) > 4:
#     utter_emb  = utter_emb[:4, :, :]
# print(torch.matmul(emo_emb, affine1).shape,torch.transpose(utter_emb, 1, 2).shape) 
# print(torch.matmul(utter_emb, affine2).shape,torch.transpose(utter_emb, 1, 2).shape)    
# A1 = F.softmax(torch.bmm(torch.matmul(emo_emb, affine1), torch.transpose(utter_emb, 1, 2)), dim=-1)
# print("A1.shape",A1.shape)
# A2 = F.softmax(torch.bmm(torch.matmul(utter_emb,affine2), torch.transpose(emo_emb, 1, 2)), dim=-1)
# emo_emb = torch.bmm(A1, utter_emb)
# print("emo_emb.shape",emo_emb.shape)
# utter_emb = torch.bmm(A2, emo_emb)
# print("utter_emb",utter_emb.shape)
# utter_emb = torch.cat([emo_emb,utter_emb ],dim=1)
# print("utter_emb",utter_emb.shape)



# utter_emb=torch.randn(3, 7, 200)#(30, 35, 49)(30, 35, 512)
# # x = torch.ones(30, 35, 49)
# # padded = nn.ConstantPad1d((0, 512 - 49), 0)(x)
# # print(padded.shape)
# emo_emb=torch.randn(4, 5,300)
# utter_emb=torch.randn(1, 7, 200)
# padded = nn.ConstantPad1d((0, 300-200), 0)(utter_emb)
# print(padded.shape)

# import pickle

# # 用 pickle 加载 pkl 文件
# file_path = './dataset/dailydialog_processed_DD.pkl'  # 替换为你的 pkl 文件路径

# with open(file_path, 'rb') as file:
#     loaded_data = pickle.load(file)

# # 打印加载的数据
# print(len(loaded_data[0]))


import torch

# # 创建一个三维张量
# tensor_3d = torch.randn(3, 11, 5)

# # 选择要固定的长度
# desired_length = 10

# # 如果第一个维度的长度大于所需长度，可以截取
# tensor_fixed = tensor_3d[:, :desired_length, :]

# # 如果第一个维度的长度小于所需长度，可以填充
# # 这里使用了随机数填充，你可以选择其他填充策略
# if tensor_3d.shape[1] < desired_length:
#     padding_size = desired_length - tensor_3d.shape[1]
#     padding_values = torch.randn(tensor_3d.shape[0],padding_size, tensor_3d.shape[2])
#     padding_values=padding_values.to(x.device)
#     tensor_fixed = torch.cat([tensor_3d, padding_values], dim=1)

# print("Original tensor:\n", tensor_3d.shape)
# print("\nTensor with the first dimension fixed:\n", tensor_fixed.shape)

# class CrossAttentionModel(nn.Module):
#     def __init__(self, feature_dim):
#         super(CrossAttentionModel, self).__init__()
        
#         # Linear mappings for query, key, and value for input1
#         self.linear_query1 = nn.Linear(feature_dim, feature_dim)
#         self.linear_key1 = nn.Linear(feature_dim, feature_dim)
#         self.linear_value1 = nn.Linear(feature_dim, feature_dim)
        
#         # Linear mappings for query, key, and value for input2
#         self.linear_query2 = nn.Linear(feature_dim, feature_dim)
#         self.linear_key2 = nn.Linear(feature_dim, feature_dim)
#         self.linear_value2 = nn.Linear(feature_dim, feature_dim)
        
#     def forward(self, input1, input2):
#         # Map input vectors to query, key, and value for input1
#         query1 = self.linear_query1(input1)
#         key1 = self.linear_key1(input1)
#         value1 = self.linear_value1(input1)
        
#         # Map input vectors to query, key, and value for input2
#         query2 = self.linear_query2(input2)
#         key2 = self.linear_key2(input2)
#         value2 = self.linear_value2(input2)
        
#         # Calculate attention weights
#         attention_weights = torch.matmul(query1, key2.transpose(1, 2))  # 交换 key2 的维度
#         attention_weights = torch.softmax(attention_weights, dim=-1)

# # Apply attention weights to values of input2
#         attended_values = torch.matmul(attention_weights, value2)

# # Integrate the attended values
#         output = torch.sum(attended_values, dim=1)
        
#         return output

# # Example usage
# feature_dim = 300
# input1 = torch.randn(4, 10, feature_dim)  # Example input1 with shape (batch_size, sequence_length, feature_dim)
# input2 = torch.randn(4, 15, feature_dim)  # Example input2 with shape (batch_size, sequence_length, feature_dim)

# model = CrossAttentionModel(feature_dim=feature_dim)
# output = model(input1, input2)

# print("Output shape:", output.shape)


class CrossAttentionModel(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttentionModel, self).__init__()
        
        # Linear mappings for query, key, and value for input1
        self.linear_query1 = nn.Linear(feature_dim, feature_dim)
        self.linear_key1 = nn.Linear(feature_dim, feature_dim)
        self.linear_value1 = nn.Linear(feature_dim, feature_dim)
        
        # Linear mappings for query, key, and value for input2
        self.linear_query2 = nn.Linear(feature_dim, feature_dim)
        self.linear_key2 = nn.Linear(feature_dim, feature_dim)
        self.linear_value2 = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, input1, input2):
        # Map input vectors to query, key, and value for input1
        query1 = self.linear_query1(input1)
        key1 = self.linear_key1(input1)
        value1 = self.linear_value1(input1)
        
        # Map input vectors to query, key, and value for input2
        query2 = self.linear_query2(input2)
        key2 = self.linear_key2(input2)
        value2 = self.linear_value2(input2)
        
        # Calculate attention weights
        attention_weights = torch.matmul(query1, key2.transpose(1, 2))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Apply attention weights to values of input2
        attended_values = torch.matmul(attention_weights, value2)
        
        # Integrate the attended values
        output = torch.sum(attended_values, dim=1)
        
        return output

# Example usage
feature_dim = 300
input1 = torch.randn(4, 10, feature_dim)  # Example input1 with shape (batch_size, sequence_length, feature_dim)
input2 = torch.randn(4, 15, feature_dim)  # Example input2 with shape (batch_size, sequence_length, feature_dim)

model = CrossAttentionModel(feature_dim=feature_dim)
output = model(input1, input2)

print("Output shape:", output.shape)
