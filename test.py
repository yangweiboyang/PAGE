import torch
import torch.nn as nn
import torch.nn.functional as F
"""
class WordRelationAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WordRelationAttention, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_text):
        # 计算注意力权重
        attention_scores = torch.matmul(input_text, self.W) + self.b
        attention_weights = F.softmax(attention_scores, dim=1)

        # 使用注意力权重加权表示句子
        weighted_input = torch.sum(input_text * attention_weights, dim=1)

        # 输出层
        output = self.linear(weighted_input)

        return output




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaPath2Vec, GATConv

python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalGraphConv(nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RelationalGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        # 定义权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(num_relations, in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        # 初始化权重
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adjacency_matrix):
        # input: (batch_size, num_nodes, in_features)
        # adjacency_matrix: (batch_size, num_relations, num_nodes, num_nodes)

        # 执行图卷积操作
        output = torch.einsum('brmn,binm->brni', [adjacency_matrix, input])  # 对应多个关系的邻接矩阵和输入特征矩阵做乘法
        output = torch.einsum('brni,rio->brno', [output, self.weight])  # 对应权重矩阵进行乘法
        output = torch.sum(output, dim=1) + self.bias  # 对所有关系的计算结果求和并加上偏置

        return F.relu(output)


class MDPGraphConvolution(nn.Module):
    def __init__(self, num_entities, num_mentions, hidden_dim, num_heads):
        super(MDPGraphConvolution, self).__init__()
        
        # 构建提及、实体和最短依存路径的节点表示
        self.mention_embedding = nn.Embedding(num_mentions, hidden_dim)
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.mdp_embedding = nn.Linear(hidden_dim, hidden_dim)
        
        # 构建多头图卷积
        self.gat_conv = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=num_heads, concat=True)
        
    def forward(self, mention_nodes, entity_nodes, mdp_nodes, edge_index):
        # 获取节点的嵌入表示
        mention_embedding = self.mention_embedding(mention_nodes)
        entity_embedding = self.entity_embedding(entity_nodes)
        mdp_embedding = self.mdp_embedding(mdp_nodes)
        
        # 构建节点特征矩阵
        x = torch.cat([mention_embedding, entity_embedding, mdp_embedding], dim=0)
        
        # 多头图卷积更新节点表示
        x = self.gat_conv(x, edge_index)
        
        return x
 """   
"""
#*********************************DenseNet网络结构***************************
# 定义DenseNet中的卷积块
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)  # 在通道维度上将输入x和输出out拼接
        return out

# 定义Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(BottleneckBlock(in_channels, growth_rate))
            in_channels += growth_rate
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

# 定义Transition层
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.avgpool(out)
        return out

# 定义DenseNet模型
class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_classes=1000):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # 初始的卷积和池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 构建Dense Block和Transition层
        num_features = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, num_layers, growth_rate)
            self.features.add_module(f'denseblock_{i+1}', block)
            num_features = num_features + num_layers*growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition_{i+1}', trans)
                num_features = num_features // 2
        
        # 最后的BN层和分类层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 创建DenseNet模型
model = DenseNet(growth_rate=32, block_config=[6, 12, 24, 16], num_classes=756)
input_data = torch.randn(1, 3, 224, 224)
output = model(input_data)
# 打印模型结构
print(output.shape)


#******************************双gat网络结构**************************
class RGAT(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout = 0.2, alpha = 0.2, nheads = 2, num_relation=-1):
        
        super(RGAT, self).__init__()
        self.dropout = dropout
    
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, relation = True, num_relation=num_relation) for _ in range(nheads)] # 多头注意力
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False, relation = True, num_relation=num_relation) # 恢复到正常维度
        
        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1) # (B,N,num_head*N_out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.gelu(self.out_att(x, adj))  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x
    def get_semantic_adj(self, speakers, max_dialog_len):
  
        semantic_adj = []
        for speaker in speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)): # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        if i==j:
                            s[i,j] = 1  # 对角线  self
                        elif i < j:
                            s[i,j] = 2   # self-future
                        else:
                            s[i,j] =3    # self-past
                    else:
                        if i<j:
                            s[i,j] = 4   # inter-future
                        elif i>j:
                            s[i,j] = 5   # inter-past
                        

            semantic_adj.append(s)
        
        return torch.stack(semantic_adj)

    #结构边

    def get_structure_adj(self, links, relations, lengths, max_dialog_len):
        '''
        map_relations = {'Comment': 0, 'Contrast': 1, 'Correction': 2, 'Question-answer_pair': 3, 'QAP': 3, 'Parallel': 4, 'Acknowledgement': 5,
                     'Elaboration': 6, 'Clarification_question': 7, 'Conditional': 8, 'Continuation': 9, 'Result': 10, 'Explanation': 11,
                     'Q-Elab': 12, 'Alternation': 13, 'Narration': 14, 'Background': 15}

        '''
        structure_adj = []

        for link,relation,length in zip(links,relations,lengths):  
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N） 0 表示填充部分 或 没有关系
            assert len(link)==len(relation)

            for index, (i,j) in enumerate(link):
                s[i,j] = relation[index] + 1
                s[j,i] = s[i,j]   # 变成对称矩阵
        
            

            for i in range(length):  # 填充对角线
                s[i,i] = 17

            structure_adj.append(s)
        
        return torch.stack(structure_adj)

class DualGATs(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        
        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        SpkGAT = []
        DisGAT = []
        for _ in range(args.gnn_layers):
            SpkGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=6))
            DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=18))

        self.SpkGAT = nn.ModuleList(SpkGAT)
        self.DisGAT = nn.ModuleList(DisGAT)


        self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        self.diff_loss = DiffLoss(args)
        self.beta = 0.3

        in_dim = args.hidden_dim *2 + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.drop = nn.Dropout(args.dropout)

       

    def forward(self, utterance_features, semantic_adj, structure_adj):
        '''
        :param tutterance_features: (B, N, emb_dim)
        :param xx_adj: (B, N, N)
        :return:
        '''
        batch_size = utterance_features.size(0)
        H0 = F.relu(self.fc1(utterance_features)) # (B, N, hidden_dim)
        H = [H0]
        diff_loss = 0
        for l in range(self.args.gnn_layers):
            if l==0:
                H1_semantic = self.SpkGAT[l](H[l], semantic_adj)
                H1_structure = self.DisGAT[l](H[l], structure_adj)
            else:
                H1_semantic = self.SpkGAT[l](H[2*l-1], semantic_adj)
                H1_structure = self.DisGAT[l](H[2*l], structure_adj)


            diff_loss = diff_loss + self.diff_loss(H1_semantic, H1_structure)
            # BiAffine 

            A1 = F.softmax(torch.bmm(torch.matmul(H1_semantic, self.affine1), torch.transpose(H1_structure, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(H1_structure, self.affine2), torch.transpose(H1_semantic, 1, 2)), dim=-1)

            H1_semantic_new = torch.bmm(A1, H1_structure)
            H1_structure_new = torch.bmm(A2, H1_semantic)

            H1_semantic_out = self.drop(H1_semantic_new) if l < self.args.gnn_layers - 1 else H1_semantic_new
            H1_structure_out = self.drop(H1_structure_new) if l <self.args.gnn_layers - 1 else H1_structure_new


            H.append(H1_semantic_out)
            H.append(H1_structure_out)

        H.append(utterance_features) 

        H = torch.cat([H[-3],H[-2],H[-1]], dim = 2) #(B, N, 2*hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        logits = self.out_mlp(H)
        return logits, self.beta * (diff_loss/self.args.gnn_layers)

"""
"""
#*****************************cls池化*******************************

import numpy as np
def cls_pooling(input, filter_size):
    # 获取输入的大小和通道数
    input_height, input_width, input_channels = input.shape
    # 计算输出的大小
    output_height = input_height // filter_size
    output_width = input_width // filter_size
    # 初始化输出
    output = np.zeros((output_height, output_width, input_channels))
    # 对每个通道进行操作
    for c in range(input_channels):
        for i in range(output_height):
            for j in range(output_width):
                # 计算输出值
                output[i, j, c] = np.max(input[i*filter_size:(i+1)*filter_size, j*filter_size:(j+1)*filter_size, c])
    return output

# 示例用法
input = torch.randn(4,300,2)
filter_size = 2
output = cls_pooling(input, filter_size)
print("CLSPooling结果:", output)
"""
#******************双视角门控网络结构*************************************
# #需要确保两个输入的第二个维度的大小是相同的（即两个视图在这一维度上具有相同的长度或特征数）。
# class DualViewGate(nn.Module):
#     def __init__(self, input_size1, input_size2, hidden_size):
#         super(DualViewGate, self).__init__()
#         self.fc1 = nn.Linear(input_size1, hidden_size)
#         self.fc2 = nn.Linear(input_size2, hidden_size)
#         self.gate = nn.Linear(3600, 300)#hidden_size * 2

#     def forward(self, x1, x2):
#         # Process input from view 1
#         out1 = F.relu(self.fc1(x1))# torch.Size([4, 20, 300])
#         print("*************lin388 out1",out1.shape)
#         # Process input from view 2
#         out2 = F.relu(self.fc2(x2))#torch.Size([4, 20, 300])
#         print("*************lin391",out1.shape,out2.shape)
#         # Concatenate the processed views along the second dimension
#         concatenated = torch.cat((out1, out2), dim=2)#torch.Size([4, 20, 600])
#         print("**********394",concatenated.shape)
#         # Reshape the concatenated tensor to match the expected input for the gate
#         concatenated = concatenated.view(concatenated.size(0), -1)#torch.Size([4, 12000])
#         print("**********397",concatenated.shape)
#         # Compute gate values
#         gate_values = torch.sigmoid(self.gate(concatenated))
#         print("**********400",gate_values.shape)
#         # Expand gate_values to match the dimensions of the views
#         gate_values = gate_values.unsqueeze(2).expand_as(out1)
#         print("**********403",gate_values.shape)
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
# view1_data = torch.rand((6, 12, 300))
# view2_data = torch.rand((6, 12, 300))

# dual_view_gate = DualViewGate(view1_size, view2_size, hidden_size)
# output = dual_view_gate(view1_data, view2_data)
# print("lin425",output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualViewGate(nn.Module):
    def __init__(self, input_size_view1, input_size_view2, hidden_size):
        super(DualViewGate, self).__init__()
        self.fc_view1 = nn.Linear(input_size_view1, hidden_size)
        self.fc_view2 = nn.Linear(input_size_view2, hidden_size)
        self.gate1 = nn.Linear(hidden_size, 1)
        self.gate2 = nn.Linear(hidden_size, 1)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc_view1(x1))
        x2 = F.relu(self.fc_view2(x2))
        
        gate1 = torch.sigmoid(self.gate1(x1))
        gate2 = torch.sigmoid(self.gate2(x2))
        
        x1_weighted = x1 * gate1
        x2_weighted = x2 * gate2
        # x1_weighted = x1_weighted.expand(-1, 20, -1)
        combined = x1_weighted + x2_weighted
        return combined

# 使用示例
input_size_view1 = 300  # 输入视图1的特征维度
input_size_view2 = 200  # 输入视图2的特征维度
hidden_size = 300  # 隐藏层的维度
batch_size = 6  # 批量大小
seq_length = 20  # 序列长度

# 生成随机的输入数据
input_view1_data = torch.rand(batch_size, 20, input_size_view1)
input_view2_data = torch.rand(batch_size, 20, input_size_view2)

# 初始化 DualViewGate 模型
model = DualViewGate(input_size_view1, input_size_view2, hidden_size)

# 进行前向传播
output = model(input_view1_data, input_view2_data)
print("**********lin471 output ",output.shape)

#**********************************交叉注意力********************************
# class CrossAttention(nn.Module):
#     def __init__(self, input_size1, input_size2, hidden_size):
#         super(CrossAttention, self).__init__()
#         self.query_fc = nn.Linear(input_size1, hidden_size)
#         self.key_fc = nn.Linear(input_size2, hidden_size)
#         self.value_fc = nn.Linear(input_size2, hidden_size)

#     def forward(self, x1, x2):
#         # Compute query, key, and value
#         query = self.query_fc(x1)
#         key = self.key_fc(x2)
#         value = self.value_fc(x2)

#         # Calculate attention scores
#         scores = torch.matmul(query, key.transpose(-2, -1))
#         attention_weights = F.softmax(scores, dim=-1)

#         # Apply attention to values
#         attended_values = torch.matmul(attention_weights, value)

#         return attended_values

# # Example usage:
# input_size1 = 300
# input_size2 = 300
# hidden_size = 300
# sequence_length1 = 20
# sequence_length2 = 15

# # Create synthetic data
# data1 = torch.rand((3, sequence_length1, 300))
# data2 = torch.rand((3, sequence_length2, 300))

# # Create CrossAttention model
# cross_attention = CrossAttention(input_size1,input_size2, hidden_size)

# # Apply CrossAttention to the data
# output = cross_attention(data1, data2)
# print(output.shape)



#******************************多专家网络结构**************
# import torch
# import torch.nn as nn

# class Expert(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Expert, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)

# class MOEModel(nn.Module):
#     def __init__(self, input_dim, expert_output_dim, gate_output_dim):
#         super(MOEModel, self).__init__()

#         # Define three experts (you can adjust the number of experts based on your needs)
#         self.expert1 = Expert(input_dim, expert_output_dim)
#         self.expert2 = Expert(input_dim, expert_output_dim)
#         self.expert3 = Expert(input_dim, expert_output_dim)

#         # Define the gate network
#         self.gate_network = nn.Sequential(
#             nn.Linear(3 * expert_output_dim, gate_output_dim),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, input_tensor):
#         # Forward pass through each expert
#         output1 = self.expert1(input_tensor)
#         output2 = self.expert2(input_tensor)
#         output3 = self.expert3(input_tensor)

#         # Concatenate expert outputs along the last dimension
#         expert_outputs = torch.cat([output1, output2, output3], dim=-1)

#         # Calculate gate weights
#         gate_weights = self.gate_network(expert_outputs)

#         # Weighted sum of expert outputs based on gate weights
#         final_output = torch.sum(expert_outputs * gate_weights, dim=-1)

#         return final_output

# # Example usage
# input_dim = 200
# expert_output_dim = 200
# gate_output_dim = 600

# # Create an instance of the MOEModel
# moe_model = MOEModel(input_dim, expert_output_dim, gate_output_dim)

# # Generate example input tensor
# input_tensor = torch.randn(4, 8, input_dim)  # Example input with shape (batch_size, sequence_length, input_dim)

# # Forward pass through the MOE model
# output_tensor = moe_model(input_tensor)

# print("Output shape:", output_tensor.shape)
#********************************************************TCN网络结构**********************************
# import torch
# import torch.nn as nn

# class TemporalBlock1(nn.Module):
#     def __init__(self, input_size, output_size, kernel_size, stride, dilation):
#         super(TemporalBlock, self).__init__()
#         padding = int((kernel_size - 1) * (dilation - 1) / 2)
#         self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.conv(x))
# class TemporalBlock(nn.Module):
#     def __init__(self, input_size, output_size, kernel_size, stride, dilation):
#         super(TemporalBlock, self).__init__()
#         padding = int((kernel_size - 1) * (dilation - 1) / 2)
#         self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         return self.relu(self.conv(x))
# class BiTemporalConvNet(nn.Module):
#     def __init__(self, input_size, output_size, num_blocks, kernel_size, stride, dilation):
#         super(BiTemporalConvNet, self).__init__()

#         self.forward_blocks = nn.ModuleList()
#         self.backward_blocks = nn.ModuleList()

#         for i in range(num_blocks):
#             self.forward_blocks.append(TemporalBlock(input_size, output_size, kernel_size, stride, dilation))
#             self.backward_blocks.append(TemporalBlock(input_size, output_size, kernel_size, stride, dilation))

#     def forward(self, x):
#         # Forward pass
#         forward_output = x
#         for block in self.forward_blocks:
#             forward_output = block(forward_output)

#         # Backward pass
#         backward_output = x.flip(dims=[2])  # Reverse the input along the time dimension
#         for block in self.backward_blocks:
#             backward_output = block(backward_output)

#         backward_output = backward_output.flip(dims=[2])  # Reverse the output along the time dimension

#         # Concatenate forward and backward outputs
#         output = torch.cat([forward_output, backward_output], dim=2)

#         return output

# # Example usage
# input_size = 10
# output_size = 16
# num_blocks = 3
# kernel_size = 3
# stride = 1
# dilation = 2

# # Create synthetic data
# sequence_length =300
# batch_size = 4
# input_data = torch.randn(batch_size, input_size, sequence_length)

# # Build Bi-TCN model
# bi_tcn_model = BiTemporalConvNet(input_size, output_size, num_blocks, kernel_size, stride, dilation)

# # Forward pass
# output_data = bi_tcn_model(input_data)

# print("Input shape:", input_data.shape)
# print("Output shape:", output_data.shape)


# import torch
# import torch.nn as nn

# class CrossAttentionLayer(nn.Module):
#     def __init__(self, query_dim, key_dim, value_dim, num_heads=1, dropout_rate=0):
#         super(CrossAttentionLayer, self).__init__()

#         self.num_heads = num_heads
#         self.dropout_rate = dropout_rate

#         # Linear projections for queries, keys, and values
#         self.query_proj = nn.Linear(query_dim, query_dim)
#         self.key_proj = nn.Linear(key_dim, key_dim)
#         self.value_proj = nn.Linear(value_dim, value_dim)

#         # Multi-head linear projection
#         self.linear_out = nn.Linear(value_dim, value_dim)

#         # Dropout layer
#         self.dropout = nn.Dropout(p=self.dropout_rate)

#     def forward(self, queries, keys, values):
#         # Linear projections
#         Q = self.query_proj(queries)  # [bs, query_len, query_dim]
#         K = self.key_proj(keys)  # [bs, key_len, key_dim]
#         V = self.value_proj(values)  # [bs, value_len, value_dim]

#         # Split and concat for multi-head attention
#         Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*bs, query_len, query_dim/h)
#         K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*bs, key_len, key_dim/h)
#         V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*bs, value_len, value_dim/h)

#         # Calculate attention scores
#         attention_scores = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*bs, query_len, key_len)
#         attention_scores = attention_scores / (K_.size()[-1] ** 0.5)  # Scale

#         # Softmax to get attention weights
#         attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

#         # Apply dropout
#         attention_weights = self.dropout(attention_weights)

#         # Weighted sum using attention weights
#         output = torch.bmm(attention_weights, V_)  # (h*bs, query_len, value_dim/h)

#         # Concatenate and project back to original dimension
#         output = torch.cat(torch.chunk(output, self.num_heads, dim=0), dim=2)  # (bs, query_len, value_dim)
#         output = self.linear_out(output)  # Linear projection

#         return output

# # Example usage
# query_dim = 256
# key_dim = 256
# value_dim = 256
# num_heads = 4

# cross_attention_layer = CrossAttentionLayer(query_dim, key_dim, value_dim, num_heads)

# # Example input shapes
# queries = torch.rand((32, 10, query_dim))  # Batch size: 32, Sequence length: 10
# keys = torch.rand((32, 15, key_dim))  # Batch size: 32, Sequence length: 15
# values = torch.rand((32, 15, value_dim))  # Batch size: 32, Sequence length: 15

# # Forward pass
# output = cross_attention_layer(queries, keys, values)

# print("Output shape:", output.shape)


# import torch
# import torch.nn as nn

# class BiLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super(BiLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 乘以2是因为双向LSTM会将正向和逆向的隐藏状态拼接在一起
#         self.w_omiga = torch.randn(input_dim,2*hidden_dim,1,requires_grad=True)#batchsize=4
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始化正向和逆向的隐藏状态
#         c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始化正向和逆向的细胞状态

#         out, _ = self.bilstm(x, (h0, c0))  # 获取双向LSTM的输出
#         # print("**********lin81 out",out.shape)#torch.Size([4, 11, 1536])
#         H = torch.nn.Tanh()(out)
#         self.w_omiga = torch.randn(H.shape[0],2*self.hidden_dim,1,requires_grad=True).to(x.device)
#         # self.w_omiga=self.w_omiga
#         weights = torch.nn.Softmax(dim=-1)(torch.bmm(H,self.w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1,1,self.hidden_dim * 2)
#         # print("*******lin85 weights",weights.shape)#torch.Size([4, 11, 1536])
#         out = torch.mul(out,weights)
#         # print("************lin86 out.shape ",out.shape)#torch.Size([4, 11, 1536])
#         out = self.fc(out[:, :, :])  # 取最后一个时间步的输出作为模型的输出
        
        
#         return out

# input_size=768
# hidden_size=768
# num_layers=768
# output_size=2
# # Create BiLSTM model
# # bilstm=BiLSTM(input_size, hidden_size, num_layers, output_size)
# bilstm=BiLSTM(768, 768, 768, 2)
# # Example input sequence (batch_size=32, sequence_length=20)
# input_sequence = torch.randn(6,20,768)
# b=input_sequence.shape[2]

# output = bilstm(input_sequence)
# print(output.shape,b)


# import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn import HeteroGraphConv
# import numpy as np

# class HeterogeneousGraphConvolution(nn.Module):
#     def __init__(self, in_feats, out_feats, rel_names):
#         super(HeterogeneousGraphConvolution, self).__init__()
#         self.conv = HeteroGraphConv({rel: nn.Linear(in_feats, out_feats) for rel in rel_names})
        
#     def forward(self, graph, inputs):
#         return self.conv(graph, inputs)

# class HeterogeneousTensorFusion(nn.Module):
#     def __init__(self, num_node_types, input_dims, hidden_dim):
#         super(HeterogeneousTensorFusion, self).__init__()
#         self.embeddings = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for input_dim in input_dims])
#         self.conv = HeterogeneousGraphConvolution(hidden_dim, hidden_dim, graph.etypes)
#         self.fc = nn.Linear(hidden_dim, 1)
        
#     def forward(self, graph, input_data):
#         print("*************lin740",input_data.shape)
#         node_embeds = [embedding(data) for embedding, data in zip(self.embeddings, input_data)]
#         node_embeds = {f'node_type_{i}': embed for i, embed in enumerate(node_embeds)}
#         h = self.conv(graph, node_embeds)
#         h = F.relu(h['node'])
#         h = dgl.mean_nodes(graph, h)
#         return self.fc(h)

# # 创建异构图
# graph = dgl.heterograph({
#     ('node_type_0', 'edge_type_0', 'node_type_1'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
#     ('node_type_1', 'edge_type_1', 'node_type_2'): (torch.tensor([2, 3]), torch.tensor([3, 4]))
# })

# # 创建三个不同类型的三维张量
# tensor_data_0 = torch.randn(2, 3, 4)  # 2 nodes of type 0, each with a 3x4 tensor
# tensor_data_1 = torch.randn(3, 3, 4)  # 3 nodes of type 1, each with a 3x4 tensor
# tensor_data_2 = torch.randn(2, 3, 4)  # 2 nodes of type 2, each with a 3x4 tensor

# # 将三维张量转换成节点特征矩阵
# node_data_0 = tensor_data_0.view(tensor_data_0.size(0), -1)
# node_data_1 = tensor_data_1.view(tensor_data_1.size(0), -1)
# node_data_2 = tensor_data_2.view(tensor_data_2.size(0), -1)

# # 构建异构图融合模型
# fusion_model = HeterogeneousTensorFusion(num_node_types=3, input_dims=[tensor_data_0.size(1), tensor_data_1.size(1), tensor_data_2.size(1)], hidden_dim=64)

# # 模型前向传播
# logits = fusion_model(graph, [node_data_0, node_data_1, node_data_2])





# class DAGCN(nn.Module):
#     """
#     DAGCN module operated on graph
#     """

#     def __init__(self):
#         super(DAGCN, self).__init__()
#         # self.args = args
#         self.in_dim = 768
#         self.num_layers = 8
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
    
# s=DAGCN()##torch.Size([4, 330/x, 768]) torch.Size([4, 330/x, 330/x]),输出torch.Size([4, 330/x, 768]) 
# f=torch.randn(4,300,768)
# f2=torch.randn(4, 300, 300)
# s1=s(f,f2)
# print("********",s1.shape)