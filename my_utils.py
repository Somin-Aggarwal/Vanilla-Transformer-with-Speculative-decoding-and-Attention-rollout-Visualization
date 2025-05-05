import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self,in_features,intermediate_features,out_features):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features,intermediate_features),
                                 nn.ReLU(),
                                 nn.Linear(intermediate_features,out_features))
        
    def forward(self, input):
        return self.mlp(input)

def scaled_dot_product_attention(query, key, value, get_attention_maps=False):
    # shape of query key value is of (batch,tokens,edim)
    output = torch.bmm(query,torch.permute(key,dims=[0,2,1]))
    output *= value.shape[1]
    output = torch.softmax(output,dim=1)
    if get_attention_maps:
        attention_maps = output
    output = torch.bmm(output,value)
    if get_attention_maps:
        return output, attention_maps
    return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, no_of_heads, edim, get_attention_maps=False):
        super().__init__()
        self.heads = no_of_heads
        self.edim = edim
        assert (edim / no_of_heads) % 1.0 == 0 
        self.qkv = nn.Linear(edim,3*edim)
        self.q = nn.Linear(edim,edim)
        self.k = nn.Linear(edim,edim)
        self.v = nn.Linear(edim,edim)
        self.linear = nn.Linear(edim, edim)
        self.get_attention_maps = get_attention_maps
        
    def forward(self, input):
        # Input is of shape (batch , tokens, edim)
        assert input.ndim == 3
        assert self.edim == input.shape[2]
        
        # Make q,k,v
        qkv = self.qkv(input) # (b,tokens, 3*edim)
        qkv = torch.reshape(qkv,shape=(qkv.shape[0],qkv.shape[1],3,self.heads,self.edim//self.heads)) # (batch, tokens,3,heads,edim//heads)
        qkv = torch.permute(qkv,dims=[2,0,3,1,4]) # (3, batch, heads, tokens, edim//heads)
        # q = (batch,heads,tokens,edim//heads) kT = (batch,heads,edim//heads,tokens) v = (batch,heads,tokens,edim//heads)
        q, kT, v = qkv[0], torch.permute(qkv[1],dims=[0,1,3,2]), qkv[2] 
        
        # Matmul Q,KT and Scale the output by 1/sqrt(dk)
        output = torch.matmul(q,kT) / math.sqrt(q.shape[-1])
        # Apply Softmax to this
        output = torch.softmax(output,dim=2)
        # matmul this to V
        output = torch.matmul(output,v)
        output = torch.reshape(output.transpose(1,2),shape=(q.shape[0],q.shape[2], q.shape[1]*q.shape[3]))
        return self.linear(output)
    
class MultiHeadAttention2(nn.Module):
    def __init__(self, no_of_heads, edim, get_attention_maps=False):
        super().__init__()
        self.heads = no_of_heads
        self.edim = edim
        assert (edim / no_of_heads) % 1.0 == 0 
        self.qkv = nn.Linear(edim,3*edim)
        self.q = nn.Linear(edim,edim)
        self.k = nn.Linear(edim,edim)
        self.v = nn.Linear(edim,edim)
        self.linear = nn.Linear(edim, edim)
        self.get_attention_maps = get_attention_maps
        
    def forward(self, input):
        # Input is of shape (batch , tokens, edim)
        assert input.ndim == 3
        assert self.edim == input.shape[2]
        
        # Make q,k,v
        qkv = self.qkv(input) # (b,tokens, 3*edim)
        qkv = torch.reshape(qkv,shape=(qkv.shape[0],qkv.shape[1],3,self.heads,self.edim//self.heads)) # (batch, tokens,3,heads,edim//heads)
        qkv = torch.permute(qkv,dims=[2,0,3,1,4]) # (3, batch, heads, tokens, edim//heads)
        # q = (batch,heads,tokens,edim//heads) kT = (batch,heads,edim//heads,tokens) v = (batch,heads,tokens,edim//heads)
        q, kT, v = qkv[0], torch.permute(qkv[1],dims=[0,1,3,2]), qkv[2] 
        
        # Matmul Q,KT and Scale the output by 1/sqrt(dk)
        output = torch.matmul(q,kT) / math.sqrt(q.shape[-1])
        # Apply Softmax to this
        output = torch.softmax(output,dim=2)
        # matmul this to V
        output = torch.matmul(output,v)
        output = torch.reshape(output.transpose(1,2),shape=(q.shape[0],q.shape[2], q.shape[1]*q.shape[3]))
        return self.linear(output)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, no_of_heads, edim, get_attention_maps=False):
        super().__init__()
        self.heads = no_of_heads
        self.edim = edim
        assert (edim / no_of_heads) % 1.0 == 0 
        self.q = nn.Linear(edim, edim)
        self.kv = nn.Linear(edim, 2*edim)
        self.linear = nn.Linear(edim, edim)
        self.get_attention_maps = get_attention_maps
        
    def forward(self, input, encoded_features):
        # Input is of shape (batch , tokens, edim)
        assert input.ndim == 3
        assert self.edim == input.shape[2]
        
        # Make q,k,v
        q = self.q(input) # (b,tokens, edim)
        q = torch.reshape(q,shape=(q.shape[0],q.shape[1],self.heads,self.edim//self.heads)) # (batch,tokens,heads,edim//heads)
        # q = (batch,heads,tokens,edim//heads)
        q = torch.permute(q,dims=[0,2,1,3])
        
        kv = self.kv(encoded_features) # (batch , tokens, 2*edim)
        kv = torch.reshape(kv,shape=(kv.shape[0],kv.shape[1],2,self.heads,self.edim//self.heads)) # (batch,tokens,2,heads,edim//heads)
        kv = torch.permute(kv,dims=[2,0,3,1,4]) # (2, batch, heads, tokens, edim//heads)
        # kT = (batch,heads,edim//heads,tokens) v = (batch,heads,tokens,edim//heads)
        kT, v = torch.permute(kv[0],dims=[0,1,3,2]), kv[1] 
        
        # Matmul Q,KT and Scale the output by 1/sqrt(dk)
        output = torch.matmul(q,kT) / math.sqrt(q.shape[-1])
        # Apply Softmax to this
        output = torch.softmax(output,dim=2)
        # matmul this to V
        output = torch.matmul(output,v)
        output = torch.reshape(output.transpose(1,2),shape=(q.shape[0],q.shape[2], q.shape[1]*q.shape[3]))
        return self.linear(output)
    
class PositionalEncodings(nn.Module):
    def __init__(self,edim: int, context_length: int, dropout: float):
        super.__init__()
        self.edim = edim
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout)
        
        positional_embedding = torch.zeros(context_length,edim)
        position = torch.arange(0,context_length,dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0,edim,2).float()*(math.log(-10000.0) / edim))
        
        positional_embedding[:, 0::2] = torch.sin(position*division_term)
        positional_embedding[0, 1::2] = torch.cos(position*division_term)

        positional_embedding.unsqueeze(0)
        
        self.register_buffer('positional_embeddings', positional_embedding)
        self.pe = positional_embedding
            
    def forwared(self,x):
        x = x + self.pe[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)
            
if __name__ == "__main__":
    dummy_input = torch.rand(size=(3,100,512))
    heads = 8
    device='cuda'
    
    MHA = MultiHeadAttention(no_of_heads=8,edim=512).to(device)
    ffn = MLP(in_features=512,intermediate_features=2048,out_features=512).to(device)
    MHCA = MultiHeadCrossAttention(no_of_heads=8,edim=512).to(device)
    # output = MHA(dummy_input)
    # output = ffn(output)
    # output = MHCA(output, output)
    # print(output.shape)
    modules = [MHA,ffn]
    for module in modules:
        print(module)
        summary(module,(100,512),batch_size=3)