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

import torch
import torch.nn as nn

# class CrossAttentionLayer(nn.Module):
#     def __init__(self, input_size):
#         super(CrossAttentionLayer, self).__init__()

#         # Query、Key和Value的线性映射
#         self.query_linear = nn.Linear(input_size, input_size)
#         self.key_linear = nn.Linear(input_size, input_size)
#         self.value_linear = nn.Linear(input_size, input_size)

#         # 注意力权重的缩放因子
#         self.scale_factor = torch.sqrt(torch.FloatTensor([input_size]))

#     def forward(self, input_query, input_key, input_value):
#         # 对Query、Key和Value进行线性映射
#         query = self.query_linear(input_query)
#         key = self.key_linear(input_key)
#         value = self.value_linear(input_value)

#         # 计算注意力权重
#         attention_weights = torch.matmul(query, key.transpose(1, 2)) / self.scale_factor

#         # 使用softmax函数获得标准化的注意力权重
#         attention_weights = torch.softmax(attention_weights, dim=-1)

#         # 应用注意力权重到Value上
#         output = torch.matmul(attention_weights, value)

#         return output

# # 示例数据
# input_size = 300
# seq_length = 10
# batch_size = 4

# # 初始化模型
# cross_attention = CrossAttentionLayer(input_size)

# # 生成示例输入
# input_query = torch.randn(batch_size, seq_length, input_size)
# input_key = torch.randn(batch_size, seq_length, input_size)
# input_value = torch.randn(batch_size, seq_length, input_size)

# # 前向传播
# output = cross_attention(input_query, input_key, input_value)
# print(output.shape)
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
#         attention_weights = torch.matmul(query1, key2.transpose(1, 2))
#         attention_weights = torch.softmax(attention_weights, dim=-1)
        
#         # Apply attention weights to values of input2
#         attended_values = torch.matmul(attention_weights, value2)
        
#         # Integrate the attended values
#         output = torch.sum(attended_values, dim=1)
        
#         return output

# # Example usage
# feature_dim = 300
# input1 = torch.randn(4, 10, feature_dim)  # Example input1 with shape (batch_size, sequence_length, feature_dim)
# input2 = torch.randn(4, 15, feature_dim)  # Example input2 with shape (batch_size, sequence_length, feature_dim)

# model = CrossAttentionModel(feature_dim=feature_dim)
# output = model(input1, input2)

# print("Output shape:", output.shape)

# """#******************双视角门控网络结构*************************************
# #需要确保两个输入的第二个维度的大小是相同的（即两个视图在这一维度上具有相同的长度或特征数）。
# class DualViewGate(nn.Module):
#     def __init__(self, input_size1, input_size2, hidden_size):
#         super(DualViewGate, self).__init__()
#         self.fc1 = nn.Linear(input_size1, hidden_size)
#         self.fc2 = nn.Linear(input_size2, hidden_size)
#         self.gate = nn.Linear(hidden_size * 2, 300)

#     def forward(self, x1, x2):
#         # Process input from view 1
#         out1 = F.relu(self.fc1(x1))# torch.Size([4, 20, 300])

#         # Process input from view 2
#         out2 = F.relu(self.fc2(x2))#torch.Size([4, 20, 300])
#         # print("*************lin391",out1.shape,out2.shape)
#         # Concatenate the processed views along the second dimension
#         concatenated = torch.cat((out1, out2))#torch.Size([4, 20, 600]) , dim=2
#         # print("**********394",concatenated.shape)
#         # Reshape the concatenated tensor to match the expected input for the gate
#         concatenated = concatenated.view(concatenated.size(0), -1)#torch.Size([4, 12000])
#         # print("**********397",concatenated.shape)
#         # Compute gate values
#         gate_values = torch.sigmoid(self.gate(concatenated))
#         print("**********",gate_values.shape)
#         # Expand gate_values to match the dimensions of the views
#         gate_values = gate_values.unsqueeze(2).expand_as(out1)
#         print("**********",gate_values.shape)
#         # Apply gate values to the views
#         gated_out1 = gate_values * out1
#         gated_out2 = (1 - gate_values) * out2

#         # Output the combination of gated views
#         output = gated_out1 + gated_out2

#         return output

# # Example usage:
# view1_size = 300
# view2_size = 300
# sequence_length = 20
# hidden_size = 150
# batch_size=4
# # Create synthetic data
# view1_data = torch.rand((batch_size, sequence_length, view1_size))
# view2_data = torch.rand((batch_size, sequence_length, view2_size))

# dual_view_gate = DualViewGate(view1_size, view2_size, hidden_size)
# output = dual_view_gate(view1_data, view2_data)
# print(output.shape)



# import torch
# import torch.nn as nn
 
# class CrossAttention(nn.Module):
#     def __init__(self, input_dim_a, input_dim_b, hidden_dim):
#         super(CrossAttention, self).__init__()
 
#         self.linear_a = nn.Linear(input_dim_a, hidden_dim)
#         self.linear_b = nn.Linear(input_dim_b, hidden_dim)
 
#     def forward(self, input_a, input_b):
#         # 线性映射
#         mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
#         mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
#         y = mapped_b.transpose(1, 2)
 
#         # 计算注意力权重
#         scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
#         attentions_a = torch.softmax(scores, dim=-1)  # 在维度2上进行softmax，归一化为注意力权重 (batch_size, seq_len_a, seq_len_b)
#         attentions_b = torch.softmax(scores.transpose(1, 2), dim=-1)  # 在维度1上进行softmax，归一化为注意力权重 (batch_size, seq_len_b, seq_len_a)
 
#         # 使用注意力权重来调整输入表示
#         # print(attentions_a.shape,input_b.shape)
#         output_a = torch.matmul(attentions_b.transpose(1,2), input_b)  # (batch_size, seq_len_a, input_dim_b)
#         output_b = torch.matmul(attentions_a.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)
#         # print("299",output_a.shape,output_b.shape)#torch.Size([4, 20, 300]) torch.Size([4, 10, 200])
        
#         # out = torch.cat((output_a,output_b),dim=2)
#         out=output_a+output_b
#         return out
 
 
# # 准备数据
# input_a = torch.randn(4, 20, 200)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
# input_b = torch.randn(4, 10, 300)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)
# # 定义模型
# input_dim_a = input_a.shape[-1]
# print("********lin308",input_dim_a)
# input_dim_b = input_b.shape[-1]
# hidden_dim = 500
# cross_attention = CrossAttention(input_dim_a, input_dim_b, hidden_dim)
 
# # 前向传播
# output_a= cross_attention(input_a, input_b)
# print("output_a\n", output_a.shape)


# import torch
# import torch.nn as nn

# class BiGRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(BiGRUModel, self).__init__()

#         # 定义双向GRU层
#         self.bi_gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)

#         # 线性层用于输出
#         self.fc = nn.Linear(hidden_size * 2, output_size)  # 因为是双向，所以乘以2

#     def forward(self, x):
#         # 前向传播
#         out, _ = self.bi_gru(x)

#         # 将双向GRU的输出进行拼接
#         out = torch.cat((out[:, -1, :hidden_size], out[:, 0, hidden_size:]), dim=1)

#         # 通过线性层获得最终输出
#         out = self.fc(out)

#         return out

# # 示例数据
# batch_size = 32
# sequence_length = 10
# input_size = 100
# output_size = 5
# hidden_size=300
# # 初始化模型
# model = BiGRUModel(input_size, hidden_size, output_size)

# # 生成示例输入
# input_data = torch.randn(batch_size, sequence_length, input_size)

# # 前向传播
# output = model(input_data)
# print(output.shape)#输出是二维的torch.Size([32, 5])



# class DAGCN(nn.Module):
#     """
#     DAGCN module operated on graph
#     """

#     def __init__(self, m):
#         super(DAGCN, self).__init__()
#         # self.args = args
#         self.in_dim = 768
#         self.num_layers = 5
#         # self.dropout = nn.Dropout(args.dropout)

#         self.proj = nn.Linear(self.in_dim, 1)

#     def conv_l2(self):
#         conv_weights = []
#         for w in self.W:
#             conv_weights += [w.weight, w.bias]
#         return sum([x.pow(2).sum() for x in conv_weights])

#     def forward(self, feature, graph_adj):
#         B = graph_adj.size(0) # 批大小

#         preds = [] # eq8里的H
#         preds.append(feature) # 存Z

#         for l in range(self.num_layers):
#             denom = torch.diag_embed(graph_adj.sum(2))  # 度矩阵D
#             deg_inv_sqrt = denom.pow(-0.5) # D 的-1/2次方
#             deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#             deg_inv_sqrt = deg_inv_sqrt.detach()
#             adj_ = deg_inv_sqrt.bmm(graph_adj) # D 的-1/2次方 * A
#             adj_ = adj_.bmm(deg_inv_sqrt) # (D 的-1/2次方 * A * D 的-1/2次方) 即 A_l

#             feature = adj_.transpose(-1, -2).bmm(feature) # eq8里的 H_l
#             preds.append(feature) # 存到 H里
#         #
#         pps = torch.stack(preds, dim=2)  # (B, N, L+1, D)  H
#         retain_score = self.proj(pps)  # (B, N, L+1, 1)
#         retain_score0 = torch.sigmoid(retain_score).view(-1, self.num_layers + 1, 1)  # (B*N, L+1, 1) eq8里的 S

#         retain_score = retain_score0.transpose(-1, -2)  # (B* N, 1, L+1) eq8里的 S ~
#         out = retain_score.bmm(
#             pps.view(-1, self.num_layers + 1, self.in_dim))  # (B*N, 1, L+1) * (B*N, L+1, D) = (B* N, 1, D)
#         out = out.squeeze(1).view(B, -1, self.in_dim)  # (B, N, D) eq8里的 X_out

#         return out # 返回图中节点的表示
    
# dagcn=DAGCN(768)
# input_a = torch.randn(4, 20, 200)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
# input_b = torch.randn(4, 10, 300)
# f=dagcn(input_a,input_b)
# print("*********416",f.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BiTCN(nn.Module):
#     def __init__(self, input_size, hidden_size, kernel_size):
#         super(BiTCN, self).__init__()
#         self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size//2)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # 将输入的维度调整为(batch_size, input_size, sequence_length)
#         x = self.conv(x)
#         x = F.relu(x)
#         return x

# class BiGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(BiGRU, self).__init__()
#         self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

#     def forward(self, x):
#         output, _ = self.gru(x)
#         return output

# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.attn = nn.Linear(hidden_size * 2, 1)

#     def forward(self, output):
#         attn_weights = F.softmax(self.attn(output), dim=1)
#         output = torch.bmm(output.permute(1, 2, 0), attn_weights).squeeze(2)
#         return output

# class BiTCN_BiGRU_Attention(nn.Module):
#     def __init__(self, input_size, hidden_size, kernel_size, num_layers, dropout):
#         super(BiTCN_BiGRU_Attention, self).__init__()
#         self.bitcn = BiTCN(input_size, hidden_size, kernel_size)
#         self.bigru = BiGRU(hidden_size, hidden_size, num_layers, dropout)
#         self.attention = Attention(hidden_size)

#     def forward(self, x):
#         x = self.bitcn(x)
#         x = self.bigru(x)
#         x = self.attention(x)
#         return x

# # 使用示例
# input_size =300 # 输入数据的特征维度
# hidden_size =768 # 隐藏层的维度
# kernel_size =3 # TCN的卷积核大小
# num_layers =4 # GRU的层数
# dropout =0.2 # GRU的dropout率
# sequence_length =768 # 序列长度
# batch_size =6 # 批量大小
# input_data =torch.randn(6,8,300) # 输入数据

# model = BiTCN_BiGRU_Attention(input_size, hidden_size, kernel_size, num_layers, dropout)
# input_data = torch.rand(batch_size, input_size, sequence_length)
# output = model(input_data)
# print(output.shape)

import torch
import torch.nn as nn

# class CrossAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossAttention, self).__init__()
#         self.input_dim = input_dim
#         self.linear_q = nn.Linear(input_dim, input_dim)
#         self.linear_k = nn.Linear(input_dim, input_dim)
#         self.linear_v = nn.Linear(input_dim, input_dim)
        
#     def forward(self, seq_a, seq_b):
#         # 计算 query、key、value
#         query = self.linear_q(seq_a)
#         key = self.linear_k(seq_b)
#         value = self.linear_v(seq_b)
        
#         # 计算注意力权重
#         scores = torch.matmul(query, key.transpose(-2, -1))
#         attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
#         # 应用注意力权重
#         output = torch.matmul(attn_weights, value)
        
#         return output

# # 创建输入序列
# seq_a = torch.rand(6, 10, 300)  # 输入序列A，长度为10，维度为256
# seq_b = torch.rand(6, 15,300)  # 输入序列B，长度为15，维度为256

# # 创建交叉注意力模型
# cross_attn = CrossAttention(input_dim=300)

# # 运行交叉注意力模型
# output = cross_attn(seq_a, seq_b)
# print(output.shape)  # 输出形状为(1, 10, 256)
import torch

# 创建形状为(4, 8, 200)的原始张量
original_tensor = torch.randn(4, 8, 200)

# 创建形状为(4, 8, 300)的全零张量
padded_tensor = torch.zeros(original_tensor.size(0), original_tensor.size(1), 300)

# 将原始张量的值复制到新张量的前200列
padded_tensor[:, :, :200] = original_tensor

print(padded_tensor.size())  # 输出形状为(4, 8, 300)