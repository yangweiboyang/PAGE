import torch
import torch.nn as nn
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
import torch.nn.functional as F

    
class PaG(nn.Module):
    def __init__(self,window,utter_dim,num_bases,max_len,posi_dim):
        super(PaG, self).__init__()
        self.max_len = max_len
        self.posi_dim = posi_dim
        input_size=10
        hidden_size=300
        num_layers=10
        output_size=2
        self.pe_k = nn.Embedding(max_len+1, posi_dim, padding_idx=0)
        self.pe_v = nn.Embedding(max_len+1, posi_dim, padding_idx=0)
        self.window = window
        self.rel_num = self.window + 2
        self.rgcn = RGCNConv(utter_dim,utter_dim,self.rel_num,num_bases=num_bases)
        self.bilstm=BiLSTM(input_size, hidden_size, num_layers, output_size)
    
    def forward(self,x,adj_index):
        # print("******lin24",x.shape,x)
        # mm=x
        batch_size = x.shape[0]#x的形状torch.Size([bc, x, 300])
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
            out = torch.cat((out,h.unsqueeze(0)),dim=0)#形状跟x相同
        # print("********************lin53",out.shape)    
        # outp=mm    #adj_index.shape  adj_index torch.Size([4, 2, 15, 15])后两位一样
        # print("******************lin56 adj_index",adj_index.shape)
        spkear=get_semantic_adj(adj_index,self.max_len ) #说话人信息  [x,10,10]
        # print("**************lin53",spkear)
        spkear = spkear.to(torch.float)
        spkear=spkear.to(x.device) 
        spkear=self.bilstm(spkear)
        print("*************lin50 speaker",spkear.shape)
        
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
            tensor_fixed = torch.cat([spkear, padding_values], dim=1)#torch.Size([4, x, 10]
        # print("**********************lin50 spkear.shape,out.shape,adj_index.shape",spkear.shape,out.shape,adj_index.shape)
        #torch.Size([1, 10, 10])后面两个维度确定 torch.Size([1, 8, 300])最后一位300确定，第二维小于或者大于10
        #torch.Size([4, 10, 10]) torch.Size([4, 11, 300]) torch.Size([4, 2, 11, 11])第二位2是确定的，第三位和前面第二位一样
        
        out=torch.cat([out,tensor_fixed],dim=2)#torch.Size([4, x, 310])
        
        out=out[:,:,:300]
        return out,rel_emb_k,rel_emb_v

class EmotionAttentionLayer(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            num_heads: An int. Number of heads.
        '''
        super(EmotionAttentionLayer, self).__init__()

        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [bs, emo_num, feat_dim]
        # queries: A 3d Variable with shape of [bs,curr_max_win_len, feat_dim]
        # Linear projections
        Q = self.Q_proj(queries)  # [bs,curr_max_win_len, feat_dim]
        K = self.K_proj(keys)  # [bs, emo_num, feat_dim]
        V = self.V_proj(values)  # [bs, emo_num, feat_dim]
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*bs, curr_max_win_len, feat_dim/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*bs, emo_num, feat_dim/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*bs, emo_num, feat_dim/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*bs, curr_max_win_len, emo_num)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5) # (h*bs, curr_max_win_len, emo_num)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  #(h*bs, curr_max_win_len, emo_num)
        '''
        # Query Masking  图注意力部分、输出部分都会排除掉填充部分， 这里填充部分正常处理即可
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        '''
        # Dropouts 中间层的emotion attention添加dropout  输出层做预测时不加
        outputs = self.output_dropout(outputs)  # (h*bs, curr_max_win_len, emo_num)
        if last_layer == True: # head=1 直接返回置信度  此时dropout=0
            return outputs #  (bs, curr_max_win_len, emo_num)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*bs, curr_max_win_len, feat_dim/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (bs, curr_max_win_len, feat_dim)
        # Residual connection
        outputs += queries # (bs, curr_max_win_len, feat_dim)

        return outputs # (bs, curr_max_win_len, feat_dim)
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 乘以2是因为双向LSTM会将正向和逆向的隐藏状态拼接在一起
        self.w_omiga = torch.randn(input_dim,2*hidden_dim,1,requires_grad=True)#batchsize=4
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始化正向和逆向的隐藏状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始化正向和逆向的细胞状态

        out, _ = self.bilstm(x, (h0, c0))  # 获取双向LSTM的输出
        # print("**********lin81 out",out.shape)#torch.Size([4, 11, 1536])
        H = torch.nn.Tanh()(out)
        self.w_omiga = torch.randn(H.shape[0],2*self.hidden_dim,1,requires_grad=True).to(x.device)
        # self.w_omiga=self.w_omiga
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H,self.w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1,1,self.hidden_dim * 2)
        # print("*******lin85 weights",weights.shape)#torch.Size([4, 11, 1536])
        out = torch.mul(out,weights)
        # print("************lin86 out.shape ",out.shape)#torch.Size([4, 11, 1536])
        out = self.fc(out[:, :, :])  # 取最后一个时间步的输出作为模型的输出
        
        
        return out
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
 



    
