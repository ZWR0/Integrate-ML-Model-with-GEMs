import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
import mytools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        # 定义一个线性层来计算每个特征维度上的注意力权重
        self.attention_weights = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # 计算注意力权重 (batch_size, input_dim)
        attn_weights = F.softmax(self.attention_weights(x), dim=-1)
        # 对输入进行加权 (batch_size, input_dim)
        output = attn_weights * x
        return output

class Seq_Encoder(nn.Module):
    def __init__(self, seq_input, seq_output, word_num, heads=2):
        super(Seq_Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 序列嵌入
        self.embedding = nn.Embedding(word_num, seq_output).to(self.device)

        # 注意力机制
        self.trans = nn.TransformerEncoderLayer(d_model=seq_output, nhead=heads, batch_first=True, norm_first=True)

        #线性降维
        self.linear = nn.Linear(seq_output, seq_input)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(seq_input)

        # 添加自定义注意力层
        self.attn = AttentionLayer(input_dim=seq_input)

    def forward(self, seqs, mask_idx=None):
        seqs = [self.embedding(s.to(self.device)) for s in seqs]
        # 填充序列
        padded_seqs = pad_sequence(seqs, batch_first=True).to(self.device)

        out = self.trans(padded_seqs)
        # 残差连接
        out = out + padded_seqs
        #池化
        out = out.mean(dim=1)

        #线性降维
        out = self.linear(out)
        out = self.relu(out)
        out = self.batch(out)

        # 使用自定义注意力层
        out1 = self.attn(out)

        # 残差
        out = out + out1
        
        return out #(batch_size, smi_output)

class Smi_Encoder(nn.Module):
    def __init__(self, smi_input, smi_output, heads=2):
        super(Smi_Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.trans = nn.TransformerEncoderLayer(d_model=smi_input, nhead=heads, batch_first=True, norm_first=True)

        self.linear = nn.Linear(smi_input, smi_output)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(smi_output)

        # 添加自定义注意力层
        self.attn = AttentionLayer(input_dim=smi_output)
        
    def forward(self, padded_smi_embedding, mask_idx=None):

        # 注意力
        out = self.trans(padded_smi_embedding)

        # 残差连接
        out = out + padded_smi_embedding
        
        #池化
        out = out.mean(dim=1)

        #线性降维
        out = self.linear(out)
        out = self.relu(out)
        out = self.batch(out)

        # 使用自定义注意力层
        out1 = self.attn(out)

        # 残差
        out = out + out1
        
        return out #(batch_size, smi_output)

class Graph_Encoder(nn.Module):
    def __init__(self, graph_input, graph_output, heads):
        """
        初始化图编码器。
        
        参数:
        - graph_input: 输入特征维度。
        - graph_output: 输出特征维度。
        - heads: 注意力头的数量。
        """
        super(Graph_Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = GATConv(graph_input, graph_output, heads=heads)
        self.norm1 = nn.BatchNorm1d(graph_output * heads)
        self.relu1 = nn.ReLU()

        self.conv2 = GATConv(graph_output * heads, graph_output * heads, heads=heads)
        self.norm2 = nn.BatchNorm1d(graph_output * heads * 2)
        self.relu2 = nn.ReLU()

    def forward(self, data, mask_idx=None):
        """
        前向传播过程。
        
        参数:
        - data: 图数据对象。
        - mask_idx: 需要掩码的节点索引。
        返回:
        - 经过池化后的图表示。
        """
        # 确保所有输入数据都在同一设备上
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch.to(self.device)

        if mask_idx is not None:
            x[mask_idx, :] *= 0  # 将需要mask的部分置零
        
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = self.relu2(x2)

        if mask_idx is not None:
            x1[mask_idx, :] *= 0
            x2[mask_idx, :] *= 0
            
        x = torch.cat([x1, x2], dim=1)
        x_mean = pyg_nn.global_mean_pool(x, batch=batch)
        x_max = pyg_nn.global_max_pool(x, batch=batch)
        return torch.cat([x_mean, x_max], dim=1) # (batch_size, graph_input*12)

class My_model(nn.Module):
    def __init__(self, graph_input, graph_output, seq_input, seq_output, word_num, smi_input, smi_output, heads):
        super(My_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.graph_encoder = Graph_Encoder(graph_input=graph_input, graph_output=graph_output, heads=heads).to(self.device)
        self.seq_encoder = Seq_Encoder(seq_input=seq_input, seq_output=seq_output, word_num=word_num, heads=heads).to(self.device)
        self.smi_encoder = Smi_Encoder(smi_input=smi_input, smi_output=smi_output, heads=heads).to(self.device)

        combined_dim = graph_output * 12 + seq_input + smi_output + 2  # 加上温度特征
        # 添加自定义注意力层
        self.attn = AttentionLayer(input_dim=combined_dim)

        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        ).to(self.device)

    def forward(self, data, spe, seqs, mask_idx=None):
        Temp_K_norm = data.Temp_K_norm.to(self.device)
        Temp_K_norm = Temp_K_norm.view(-1, 1)
        Inv_Temp_norm = data.Inv_Temp_norm.to(self.device)
        Inv_Temp_norm = Inv_Temp_norm.view(-1, 1)

        x1 = self.graph_encoder(data)
        x2 = self.smi_encoder(spe)
        x3 = self.seq_encoder(seqs)

        out = torch.cat([x1, x2, x3, Temp_K_norm, Inv_Temp_norm], dim=1)

        # 注意力
        out1 = self.attn(out)

        # 残差
        out = out + out1
        
        out = self.relu(out)
        out = self.fc(out)
        
        return out.squeeze()