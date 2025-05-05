import torch
import torch.nn as nn
from my_utils import MultiHeadAttention,MLP, MultiHeadCrossAttention
from torchsummary import summary
import math

class Encoder(nn.Module):
    def __init__(self, no_of_stacks, no_of_heads, edim,
                 in_f,inter_f,out_f ,get_attention_maps=False):
        super().__init__()
        self.no_of_stacks = no_of_stacks
        self.heads = no_of_heads
        self.edim = edim
        self.get_attention_maps = get_attention_maps
        self.MHA = MultiHeadAttention(no_of_heads,edim,get_attention_maps)
        self.MLP = MLP(in_f, inter_f, out_f)
        self.layerNorm = nn.LayerNorm(edim)
        
    def forward(self,input):
        # Input Shape (Batch, Context Lenght, Embedding_dim)
        assert input.ndim==3
        
        output = input
        for i in range(self.no_of_stacks):
            store = output
            
            # Multi Head Attention
            output = self.MHA(output)
            # Add and Norm
            output += store
            output = self.layerNorm(output)
            
            store = output
            
            # Feed Forward
            output = self.MLP(output)
            # Add and Norm 
            output += store
            output = self.layerNorm(output)

        return output

class Decoder(nn.Module):
    def __init__(self, no_of_stacks, no_of_heads, edim,
                 in_f,inter_f,out_f ,get_attention_maps=False):
        super().__init__()
        self.no_of_stacks = no_of_stacks
        self.heads = no_of_heads
        self.edim = edim
        self.get_attention_maps = get_attention_maps
        self.MHA = MultiHeadAttention(no_of_heads,edim,get_attention_maps)
        self.MHCA = MultiHeadCrossAttention(no_of_heads,edim,get_attention_maps)
        self.MLP = MLP(in_f, inter_f, out_f)
        self.layerNorm = nn.LayerNorm(edim)
        
    def forward(self,input,encoded_features):
        '''
        The Decoders input is the output of decoder itself .
        The start token is given as input to the model at the beginning 
        '''
        # input shape (b,context lenght, embed_dim)
        output = input
        for i in range(self.no_of_stacks):
            store = output
            
            # Masked Multi Head Attention (Input will be masked in the dataset only for this implementation)
            output = self.MHA(output)
            # Add and Norm
            output += store
            output = self.layerNorm(output)
            
            store = output
            
            # Cross Multi Head Attention (query - decoder , key,value- encoded_features)
            output = self.MHCA(output, encoded_features)
            # Add and Norm
            output += store
            output = self.layerNorm(output)
            
            store = output
            
            # Feed Forward 
            output = self.MLP(output)
            # Add and Norm
            output += store
            output = self.layerNorm(output)
            
        return output

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, no_of_stacks, no_of_heads, edim,
                 in_features,intermediate_features,out_features, pad_idx, get_attention_maps=False):
        super().__init__()
        self.encoder = Encoder(no_of_stacks,no_of_heads,edim,in_features,
                      intermediate_features, out_features)
        self.decoder = Decoder(no_of_stacks,no_of_heads,edim,in_features,
                      intermediate_features, out_features)
        
        self.vocab = nn.Embedding(vocab_size,edim,padding_idx=pad_idx)
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_length, edim))
        self.last_linear = nn.Linear(edim,vocab_size)
        self.edim = edim
        
    def forward(self, encoder_input, decoder_input):
        # assuming Input of shape (Batch ,Context lenght) 
        # Converting INput to (Batch, COntext_lenght,edim) by getting the correspoding edim from dictionary
        encoder_input = self.vocab(encoder_input) * math.sqrt(self.edim)
        decoder_input = self.vocab(decoder_input) * math.sqrt(self.edim)
        
        # Add Positional Embeddings to the input of Encoder
        encoder_input += self.positional_embedding
        # Encoder
        encoded_features = self.encoder(encoder_input)  
        # Add positional Embeddings to the input of Decoder
        decoder_input += self.positional_embedding 
        # Decoder
        decoded_features = self.decoder(decoder_input, encoded_features)
        # Linear layer ( Input - Batch, last token, edim) ( Output - Batch, vocab size )
        output = self.last_linear(decoded_features[:,0])
        # Softmax (input - (batch , 1, vocab_size) or (batch, vocab_size))
            # output = torch.softmax(output,dim=-1)
        # Return the predicted Token
            # output = torch.argmax(output,dim=-1)   
        
        return output
    
    
if __name__=='__main__':
    no_of_stacks = 3
    no_of_heads = 8
    edim = 512
    in_features = edim
    out_features = edim
    intermediate_features = 2048
    device = 'cuda'
    context_length = 300
    vocab_size = 8192
    pad_idx = 255
        
    dummy_input = torch.rand(size=(3,context_length,edim)).to(device)

    transformer = Transformer(vocab_size,context_length, no_of_stacks, no_of_heads,edim,
                              in_features, intermediate_features, out_features, pad_idx).to(device)
    
    
    for name, param in transformer.named_parameters():
        print(name, param.grad is not None)

    # from torchsummary import summary
    # summary(transformer, input_size=[(context_length, edim), (context_length, edim)], batch_size=1)
    
    # vocab = nn.Embedding(8192,512)
    # a = torch.zeros(size=(1,300),dtype=torch.long)
    # b = vocab(a)
    # print(b.dtype)