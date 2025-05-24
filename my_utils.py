import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self,edim,intermediate_features,dropout:float):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(edim,intermediate_features),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(intermediate_features,edim))
        
    def forward(self, input):
        return self.mlp(input)

def scaled_dot_product_attention(query, key, value, mask, dropout : nn.Dropout):
    # shape of query key value is of (batch,h,tokens,edim)
    d_k = key.shape[-1]
    output = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k) # (Q.KT) / d_k
    
    if mask is not None:
        output.masked_fill_(mask == 0, -1e9)
    
    attention_maps = torch.softmax(output,dim=-1) # (batch,heads,tokens,tokens)
    
    if dropout is not None:
        attention_maps = dropout(attention_maps) 
        
    output = torch.matmul(attention_maps,value)
    
    return output, attention_maps
    
class MultiHeadAttention(nn.Module):
    def __init__(self, no_of_heads : int, edim : int, dropout : float):
        super().__init__()
        self.heads = no_of_heads
        self.edim = edim
        assert edim % no_of_heads == 0 
        self.d_model = edim // no_of_heads
        self.q = nn.Linear(edim,edim)
        self.k = nn.Linear(edim,edim)
        self.v = nn.Linear(edim,edim)
        self.linear = nn.Linear(edim, edim)
        self.dropout = nn.Dropout(dropout)
    
        
    def forward(self, query, key, value, mask):
        # Input is of shape (batch , tokens, edim)
        assert query.ndim == 3 
        assert key.ndim == 3
        assert value.ndim == 3
        
        # Make q,k,v
        # (Batch.tokens,edim) --> (Batch, heads, token, edim//heads)
        q = self.q(query) 
        k = self.k(key)
        v = self.v(value)                                       
        q = torch.reshape(q,shape=(q.shape[0], q.shape[1], self.heads, self.d_model)).transpose(1, 2)
        k = torch.reshape(k,shape=(k.shape[0], k.shape[1], self.heads, self.d_model)).transpose(1, 2)
        v = torch.reshape(v,shape=(v.shape[0], v.shape[1], self.heads, self.d_model)).transpose(1, 2)
        
        output , self.attention_maps = scaled_dot_product_attention(q,k,v,mask, self.dropout)
        output = output.transpose(1,2)
        output = torch.reshape(output,shape=(output.shape[0],output.shape[1],self.edim))
        return self.linear(output)
    
class PositionalEncodings(nn.Module):
    def __init__(self,edim: int, context_length: int, dropout: float):
        super().__init__()
        self.edim = edim
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout)
        
        positional_embedding = torch.zeros(context_length,edim)
        position = torch.arange(0,context_length,dtype=torch.float).unsqueeze(1) # shape (context_length,1)
        division_term = torch.exp(torch.arange(0,edim,2).float()*(-math.log(10000.0) / edim))
        
        positional_embedding[:, 0::2] = torch.sin(position*division_term)
        positional_embedding[:, 1::2] = torch.cos(position*division_term)

        positional_embedding = positional_embedding.unsqueeze(0)
        
        self.pe = positional_embedding
        self.register_buffer('positional_embeddings', positional_embedding)
            
    def forward(self,x):
        assert x.ndim == 3
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)
            
if __name__ == "__main__":
    dummy_input = torch.rand(size=(3,100,512))
    heads = 8
    device='cuda'
    
    MHA = MultiHeadAttention(no_of_heads=8,edim=512).to(device)
    ffn = MLP(in_features=512,intermediate_features=2048,out_features=512).to(device)
    # output = MHA(dummy_input)
    # output = ffn(output)
    # output = MHCA(output, output)
    # print(output.shape)
    modules = [MHA,ffn]
    for module in modules:
        print(module)
        summary(module,(100,512),batch_size=3)