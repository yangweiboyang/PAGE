
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch.nn.utils.rnn import pad_sequence
from tcn import TemporalConvNet
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
class UtterEncoder2(nn.Module):
    def __init__(self, model_size, utter_dim, conv_encoder='none', rnn_dropout=None):
        super(UtterEncoder2, self).__init__()
        encoder_path = 'roberta-' + model_size
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        if model_size == 'base':
            token_dim = 768
        else:
            token_dim = 1024
        self.mapping = nn.Linear(token_dim, utter_dim)
        if conv_encoder == 'none':
            self.register_buffer('conv_encoder', None)
        else:
            self.conv_encoder = nn.GRU(input_size=utter_dim,
                                       hidden_size=utter_dim,
                                       bidirectional=False,
                                       num_layers=1,
                                       dropout=rnn_dropout,
                                       batch_first=True)

    def forward(self, conv_utterance, attention_mask, conv_len):
        # conv_utterance: [[conv_len1, max_len1], [conv_len2, max_len2], ..., [conv_lenB, max_lenB]]
        processed_output = []
        for cutt, amsk in zip(conv_utterance, attention_mask):
            output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state
            # [conv_len, token_dim] -> [conv_len, utter_dim]
            pooler_output = torch.max(output_data, dim=1)[0]
            mapped_output = self.mapping(pooler_output)
            processed_output.append(mapped_output)
        # [batch_size, conv_size, utter_dim]
        conv_output = pad_sequence(processed_output, batch_first=True)
        if self.conv_encoder is not None:
            pad_conv = pack_padded_sequence(conv_output, conv_len, batch_first=True, enforce_sorted=False)
            pad_output = self.conv_encoder(pad_conv)[0]
            conv_output = pad_packed_sequence(pad_output, batch_first=True)[0]
        return conv_output
class UtterEncoder(nn.Module):
    def __init__(self, utter_dim, emotion_emb,emotion_dim,att_dropout,mlp_dropout,pag_dropout,ff_dim,nhead):
        super(UtterEncoder, self).__init__()
        encoder_path = 'roberta-base'
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        token_dim = 768
        self.mapping = nn.Linear(token_dim, utter_dim)

        self.emotion_embeddings = nn.Embedding(emotion_emb.shape[0], emotion_emb.shape[1], padding_idx=0, _weight=emotion_emb)
        self.emotion_lin = nn.Linear(emotion_emb.shape[1], emotion_dim)
        self.emotion_mapping = nn.Linear(utter_dim + emotion_dim, utter_dim)
        
        self.attention = MultiHeadAttention(nhead, utter_dim, att_dropout)
        self.norm = nn.LayerNorm(utter_dim)
        self.dropout = nn.Dropout(pag_dropout)
        self.mlp = MLP(utter_dim, ff_dim, mlp_dropout)
        self.tcn_net = TemporalConvNet(768, num_channels=[768])#
        self.bilstm=BiLSTM(768, 768, 768, 2)
    def forward(self, conv_utterance, attention_mask,adj,emotion_label):
        processed_output = []
        for cutt, amsk in zip(conv_utterance, attention_mask):
            output_data = self.encoder(cutt, attention_mask=amsk).last_hidden_state  
            # output_data=output_data.to(adj.device)
            # print("*************line33",output_data.shape) #torch.Size([x, x, 768])

            # output_data=self.bilstm(output_data)
            # print("**************lin73 output_data",output_data.shape)
            # output_data  = self.tcn_net(output_data.permute(0,2,1))
            # # # #output_data形状：([5, 27, 768])只有768是定值.
            # output_data =output_data.permute(0,2,1)#形状同：output_data
            # print("*************line37 tcn",output_data.shape)
            pooler_output = torch.max(output_data, dim=1)[0]  
            mapped_output = self.mapping(pooler_output)  
            processed_output.append(mapped_output)

        conv_output = pad_sequence(processed_output, batch_first=True)#torch.Size([6或者x, x, 300])  
        emo_emb = self.emotion_lin(self.emotion_embeddings(emotion_label))
        utter_emb = self.emotion_mapping(torch.cat([conv_output, emo_emb], dim=-1))#跟conv_output形状一样
        utter_emb=conv_output
        
        x = utter_emb.transpose(0,1)
        x2 = self.attention(x, adj)#跟conv_output形状一样
        ss = x.transpose(0,1) + self.dropout(x2)#跟conv_output形状一样
        # ss = self.norm(ss)#跟conv_output形状一样
        out = self.mlp(ss)#跟conv_output形状一样

        return ss


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        self.num_layers = num_layers
        self.output_dim=output_dim
        self.bilstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)  # 乘以2是因为双向LSTM会将正向和逆向的隐藏状态拼接在一起
        self.w_omiga = torch.randn(input_dim,2*hidden_dim,1,requires_grad=True)#batchsize=4
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始化正向和逆向的隐藏状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始化正向和逆向的细胞状态
        # print("*****************lin117")
        # self.bilstm=self.bilstm.to(x.device)
        out, _ = self.bilstm(x, (h0, c0))  # 获取双向LSTM的输出
        # print("**********lin81 out",out.shape)#torch.Size([4, 11, 1536])
        out=out.to(x.device)
        H = torch.nn.Tanh()(out)
        self.w_omiga = torch.randn(H.shape[0],2*self.hidden_dim,1,requires_grad=True).to(x.device)
        # self.w_omiga=self.w_omiga
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H,self.w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1,1,self.hidden_dim * 2)
        # print("*******lin85 weights",weights.shape)#torch.Size([4, 11, 1536])
        out = torch.mul(out,weights)
        # print("************lin86 out.shape ",out.shape)#torch.Size([4, 11, 1536])
        out = self.fc(out[:, :, :])  # 取最后一个时间步的输出作为模型的输出
        
        
        return out   
class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, utter_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim = utter_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.o_proj_weight = nn.Parameter(torch.empty(utter_dim, utter_dim), requires_grad=True)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.q_proj_weight)
        torch.nn.init.xavier_uniform_(self.k_proj_weight)
        torch.nn.init.xavier_uniform_(self.v_proj_weight)
        torch.nn.init.xavier_uniform_(self.o_proj_weight)

    def forward(self, x, adj):

        slen = x.size(0)
        bsz = x.size(1)
        adj = adj.unsqueeze(1).expand(bsz, self.nhead, slen, slen)
        adj = adj.contiguous().view(bsz*self.nhead, slen, slen) 
        scaling = float(self.head_dim) ** -0.5
  
        query = x
        key = x
        value = x
        

        query = query.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(2)
        key = key.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        value = value.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1).unsqueeze(1)
        
        attention_weight = query*key
        attention_weight = attention_weight.sum(3) * scaling
      
        attention_weight = mask_logic(attention_weight, adj)
        attention_weight = F.softmax(attention_weight, dim=2)

        attention_weight = F.dropout(attention_weight, p=self.dropout, training=True)

        attn_sum = (value * attention_weight.unsqueeze(3)).sum(2)
        attn_sum = attn_sum.transpose(0, 1).contiguous().view(bsz, slen, -1)
        output = F.linear(attn_sum, self.o_proj_weight)
       
        return output

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        linear_out = self.linear2(F.relu(self.linear1(x)))
        output = self.norm(self.dropout(linear_out) + x)
        return output


def mask_logic(alpha, adj):
    return alpha - (1 - adj) * 1e30

