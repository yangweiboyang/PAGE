import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Causal_Classifier,PaG,CausePredictor
from encoder import UtterEncoder


class PAGE(nn.Module):
    def __init__(self,
                 utter_dim,
                 emo_emb,
                 emotion_dim,
                 att_dropout,
                 mlp_dropout,
                 pag_dropout,
                 ff_dim,
                 nhead,
                 window,
                 num_bases,
                 max_len,
                 posi_dim,
                
                        ):
        super(PAGE, self).__init__()
        self.utter_encoder = UtterEncoder(utter_dim, emo_emb,emotion_dim,att_dropout,mlp_dropout,pag_dropout,ff_dim,nhead)
        self.pag = PaG(window,utter_dim,num_bases,max_len,posi_dim)
        self.classifier = Causal_Classifier(utter_dim, utter_dim)
        hidden_dim=100
        self.classifier1 = CausePredictor(3*hidden_dim, 3*hidden_dim, mlp_dropout=mlp_dropout)
        self.layers = 3
        self.affine1 = nn.Parameter(torch.Tensor(300,300))#1024
        self.affine2 = nn.Parameter(torch.Tensor(300,300))#1024
        self.drop = nn.Dropout(0.1)#有情感0.1，无情感0.4
        

        self.emotion_embeddings = nn.Embedding(emo_emb.shape[0], emo_emb.shape[1], padding_idx=0, _weight=emo_emb)
        self.emotion_lin = nn.Linear(emo_emb.shape[1], emotion_dim)
        self.emotion_mapping = nn.Linear(utter_dim + emotion_dim, utter_dim)
    def forward(self, input_ids, attention_mask, mask, adj,label,adj_index):
        utter_emb = self.utter_encoder(input_ids, attention_mask,adj,label) 
        emo_emb = self.emotion_lin(self.emotion_embeddings(label))#torch.Size([x, x, 200])
        # utter_emb=self.emotion_mapping(torch.cat([utter_emb,emo_emb],dim=-1))
        utter_emb,rel_emb_k,rel_emb_v = self.pag(utter_emb,adj_index)
        utter_emb=self.emotion_mapping(torch.cat([utter_emb,emo_emb],dim=-1))
        logits = self.classifier(utter_emb,rel_emb_k,rel_emb_v,mask)
        # logits = self.classifier1(utter_emb, 1, mask)
        return logits
    def forward1(self, input_ids, attention_mask, mask, adj,label,adj_index):
        utter_emb = self.utter_encoder(input_ids, attention_mask,adj,label) 
        emo_emb = self.emotion_lin(self.emotion_embeddings(label))#torch.Size([x, x, 200])        
        # utter_emb = self.emotion_mapping(torch.cat([utter_emb, emo_emb], dim=-1))#torch.Size([4, x, 300])，这行可有可无
        #做形状变化，让utter_emb和emo_emb都变成4,50,300，其中50有待确定具体值
        emo_emb = nn.ConstantPad1d((0, 300-200), 1)(emo_emb)  
  
        A1 = F.softmax(torch.bmm(torch.matmul(emo_emb, self.affine1), torch.transpose(utter_emb, 1, 2)), dim=-1)
        A2 = F.softmax(torch.bmm(torch.matmul(utter_emb, self.affine2), torch.transpose(emo_emb, 1, 2)), dim=-1)
        emo_emb = torch.bmm(A1, utter_emb)
        utter_emb = torch.bmm(A2, emo_emb)

        emo_emb = emo_emb[:, :, :200]
        utter_emb=self.emotion_mapping(torch.cat([utter_emb,emo_emb],dim=-1))

        utter_emb,rel_emb_k,rel_emb_v = self.pag(utter_emb,adj_index)#utter_emb torch.Size([4, x, 300])
        utter_emb = self.emotion_mapping(torch.cat([utter_emb,emo_emb],dim=-1))
        
        # logits = self.classifier(utter_emb,rel_emb_k,rel_emb_v,mask)
        logits = self.classifier1(utter_emb, 1, mask)
        return logits
    def forward2(self, input_ids, attention_mask, mask, adj,label):
        utter_emb = self.utter_encoder(input_ids, attention_mask,adj,label) 
        batch_size = utter_emb.size(0)
        H0 = F.relu(self.fc1(utter_emb)) # (B, N, hidden_dim)
        H = [H0]
        diff_loss = 0
        for l in range(2):
            if l==0:
                H1_semantic = self.SpkGAT[l](H[l], get_semantic_adj)#
                H1_structure = self.DisGAT[l](H[l], structure_adj)
            else:
                H1_semantic = self.SpkGAT[l](H[2*l-1], get_semantic_adj)
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
    def get_semantic_adj(adj_index, max_len):
        semantic_adj = []
        for speaker in adj_index:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_len, max_len, dtype = torch.long) # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)): # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(len(speaker)):
                # print("*********************lin59 speaker",speaker[i],speaker[j])
                    if torch.equal(speaker[i] ,speaker[j]):
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

       
        

class CrossAttention(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(CrossAttention, self).__init__()
        self.query_fc = nn.Linear(input_size1, hidden_size)
        self.key_fc = nn.Linear(input_size2, hidden_size)
        self.value_fc = nn.Linear(input_size2, hidden_size)

    def forward(self, x1, x2):
        # Compute query, key, and value
        query = self.query_fc(x1)
        key = self.key_fc(x2)
        value = self.value_fc(x2)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, value)

        return attended_values
