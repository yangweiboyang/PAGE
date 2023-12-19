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

#******************双视角门控网络结构*************************************
#需要确保两个输入的第二个维度的大小是相同的（即两个视图在这一维度上具有相同的长度或特征数）。
class DualViewGate(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(DualViewGate, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.fc2 = nn.Linear(input_size2, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, x1, x2):
        # Process input from view 1
        out1 = F.relu(self.fc1(x1))

        # Process input from view 2
        out2 = F.relu(self.fc2(x2))

        # Concatenate the processed views along the second dimension
        concatenated = torch.cat((out1, out2), dim=2)

        # Reshape the concatenated tensor to match the expected input for the gate
        concatenated = concatenated.view(concatenated.size(0), -1)

        # Compute gate values
        gate_values = torch.sigmoid(self.gate(concatenated))

        # Expand gate_values to match the dimensions of the views
        gate_values = gate_values.unsqueeze(2).expand_as(out1)

        # Apply gate values to the views
        gated_out1 = gate_values * out1
        gated_out2 = (1 - gate_values) * out2

        # Output the combination of gated views
        output = gated_out1 + gated_out2

        return output

# Example usage:
view1_size = 10
view2_size = 8
sequence_length = 20
hidden_size = 16

# Create synthetic data
view1_data = torch.rand((batch_size, sequence_length, view1_size))
view2_data = torch.rand((batch_size, sequence_length, view2_size))

dual_view_gate = DualViewGate(view1_size, view2_size, hidden_size)
output = dual_view_gate(view1_data, view2_data)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, expert_dim)
        self.fc2 = nn.Linear(expert_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, output_dim):
        super(MixtureOfExperts, self).__init__()

        # Create expert modules
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim, output_dim) for _ in range(num_experts)])

        # Create gate module
        self.gate = Gate(input_dim, num_experts)

    def forward(self, x):
        # Compute expert outputs
        expert_outputs = [expert(x) for expert in self.experts]

        # Compute gate weights
        gate_weights = self.gate(x)

        # Combine expert outputs using gate weights
        mixed_output = sum(weight * expert_output for weight, expert_output in zip(gate_weights.t(), expert_outputs))

        return mixed_output, gate_weights

# 示例用法
input_dim = 3  # 三个维度的输入
expert_dim = 20
num_experts = 5
output_dim = 3  # 三个维度的输出

model = MixtureOfExperts(input_dim, expert_dim, num_experts, output_dim)

# 输入数据（假设输入数据是三维的）
input_data = torch.randn(32, input_dim)

# 模型前向传播
output, gate_weights = model(input_data)

print("Output shape:", output.shape)
print("Gate weights shape:", gate_weights.shape)