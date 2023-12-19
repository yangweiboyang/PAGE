import torch
import torch.nn as nn
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


    
class PaG(nn.Module):
    def __init__(self,window,utter_dim,num_bases,max_len,posi_dim):
        super(PaG, self).__init__()
        self.max_len = max_len
        self.posi_dim = posi_dim
       
        self.pe_k = nn.Embedding(max_len+1, posi_dim, padding_idx=0)
        self.pe_v = nn.Embedding(max_len+1, posi_dim, padding_idx=0)
        self.window = window
        self.rel_num = self.window + 2
        self.rgcn = RGCNConv(utter_dim,utter_dim,self.rel_num,num_bases=num_bases)

    
    def forward(self,x,adj_index):
        batch_size = x.shape[0]
        x_dim = x.shape[2]
        slen = x.shape[1]
        src_pos = torch.arange(slen).unsqueeze(0)
        tgt_pos = torch.arange(slen).unsqueeze(1)
        pos_mask = (tgt_pos - src_pos) + 1
        pos_mask = pos_mask.to(x.device)
        
        position_mask = torch.clamp(pos_mask, min=0, max=self.max_len).long()
        rel_emb_k = self.pe_k(position_mask)
        rel_emb_v = self.pe_v(position_mask)
        
        rel_emb_k = rel_emb_k.unsqueeze(0).expand(batch_size, slen, slen, self.posi_dim)
        rel_emb_v = rel_emb_v.unsqueeze(0).expand(batch_size, slen, slen, self.posi_dim)

        rel_adj = (src_pos - tgt_pos).to(x.device)
        
        self.rgcn.to(x.device)

        rel_adj = rel_adj_create(rel_adj,slen,self.window)
        index = index_create(slen).to(x.device)
        
        edge_type = torch.flatten(rel_adj).long().to(x.device)

        out = self.rgcn(x[0],index,edge_type).unsqueeze(0)
        for i in range(1,batch_size):
            h = self.rgcn(x[i],index,edge_type)
            out = torch.cat((out,h.unsqueeze(0)),dim=0)
        spkear=get_semantic_adj(adj_index,self.max_len ) #说话人信息  
        spkear=spkear.to(x.device) 
        m2=out.shape[1]
        spkear= spkear[:,:m2,:]
        tensor_fixed=torch.randn(out.shape[0],out.shape[1],out.shape[2])
        tensor_fixed=tensor_fixed.to(x.device)
# 如果第一个维度的长度小于所需长度，可以填充
# 这里使用了随机数填充，你可以选择其他填充策略
        if  spkear.shape[1] < m2:
            padding_size = m2 - spkear.shape[1]
            padding_values = torch.randn(spkear.shape[0],padding_size, spkear.shape[2])
            padding_values=padding_values.to(x.device)
            tensor_fixed = torch.cat([spkear, padding_values], dim=1)
        # print("**********************lin50 spkear.shape,out.shape,adj_index.shape",spkear.shape,out.shape,adj_index.shape)
        #torch.Size([1, 10, 10])后面两个维度确定 torch.Size([1, 8, 300])最后一位300确定，第二维小于或者大于10
        #torch.Size([4, 10, 10]) torch.Size([4, 11, 300]) torch.Size([4, 2, 11, 11])第二位2是确定的，第三位和前面第二位一样
        # spkear = spkear.unsqueeze(2).expand(-1, -1, 300, -1)
        # out=out+spkear
        # out1=out.size(0)
        # out2=out.size(1)
        # out3=out.size(2)
        # spkear = spkear.expand(-1, -1, 300)
        # out=torch.cat(spkear,out)
        # out=out[:out1,:out2,:out3]
        # print("*********************lin72 out,tensor_fixed",out.shape,tensor_fixed.shape)
        # tensor_fixed = torch.cat(tensor_fixed, dim=2)
        out=torch.cat([out,tensor_fixed],dim=2)
        
        out=out[:,:,:300]
        return out,rel_emb_k,rel_emb_v

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
class Causal_Classifier(nn.Module):
    def __init__(self, input_dim, mlp_dim, mlp_dropout=0.1):
        super(Causal_Classifier, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout

        self.mlp = nn.Sequential(nn.Linear(2*input_dim+200, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, 1, False)

    def forward(self, x,rel_emb_k,rel_emb_v,mask):

        batch_size = x.shape[0]
        x_dim = x.shape[2]
        slen = x.shape[1]

        x_source = x.unsqueeze(1).expand(batch_size, slen, slen, x_dim)
        x_target = x.unsqueeze(2).expand(batch_size, slen, slen, x_dim)


        x_source = torch.cat([x_source,rel_emb_k],dim=-1)
        x_target = torch.cat([x_target,rel_emb_v],dim=-1)
        x_cat = torch.cat([x_source, x_target], dim=-1)  

        #predict_score,mask torch.Size([4, 11, 11]) torch.Size([4, 11, 11]
        predict_score = self.predictor_weight(self.mlp(x_cat)).squeeze(-1)
        # print("*********************lin94 predict_score,mask",predict_score.shape,mask.shape)
        predict_score = torch.sigmoid(predict_score) * mask
       
        return predict_score
#这个分类函数是KBCIN的
class CausePredictor(nn.Module):
    def __init__(self, input_dim, mlp_dim, mlp_dropout=0.1):
        super(CausePredictor, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, 1, False)

    def forward(self, x, conv_len, mask):
        predict_score = self.predictor_weight(self.mlp(x)).squeeze(-1)
        #lin94 predict_score,mask torch.Size([4, 21]) torch.Size([4, 21, 21])
        # print("*********************lin94 predict_score,mask",predict_score.shape,mask.shape)
        predict_score = torch.sigmoid(predict_score) * mask
        
        return predict_score
def rel_adj_create(rel_adj,slen,window):
    for i in range(slen):
        for s in range(i+1,slen):
            rel_adj[i][s] = 1
    
    for i in range(slen):
        num = 1     
        for o in range(i-1,-1,-2):
            if((o-1)<0):
                rel_adj[i][o] = -num
            else:
                rel_adj[i][o] = -num
                rel_adj[i][o-1] = -num
            num+=1
    
    for i in range(slen):
        for o in range(i-1,-1,-1):
            if(rel_adj[i][o]<-(window+1)):
                rel_adj[i][o] = - (window + 1) 
    
    return rel_adj

def index_create(slen):
    index = []
    start = []
    end = []     
    
    for i in range(0,slen):
        for j in range(0,slen):
            start.append(i)
    for i in range(0,slen):
        for j in range(0,slen):
            end.append(j)

    index.append(start)
    index.append(end)
    
    index = torch.tensor(index).long()
    
    return index
 



    
