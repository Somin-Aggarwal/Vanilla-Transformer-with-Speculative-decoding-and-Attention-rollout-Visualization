import torch
import torch.nn as nn
from my_utils import MultiHeadAttention,MLP, PositionalEncodings
from torchsummary import summary
import math

class EncoderBlock(nn.Module):
    
    def __init__(self, no_of_heads:int, edim:int, inter_ff:int, dropout:float):
        super().__init__()
        self.heads = no_of_heads
        self.edim = edim
        self.MHA = MultiHeadAttention(no_of_heads,edim,dropout)
        self.MLP = MLP(edim,inter_ff,dropout)   
        self.layerNorm1 = nn.LayerNorm(edim)
        self.layerNorm2 = nn.LayerNorm(edim)
        
    def forward(self,encoder_input, encoder_input_mask):
        # Input Shape (Batch, Context Lenght, Embedding_dim)
        assert encoder_input.ndim==3
        
        # Multi Head Attention
        output = self.MHA(encoder_input,encoder_input,encoder_input,encoder_input_mask)
        # Add and Norm
        output += encoder_input
        output = self.layerNorm1(output)
        
        store = output
        
        # Feed Forward
        output = self.MLP(output)
        # Add and Norm 
        output += store
        output = self.layerNorm2(output)

        return output

class Encoder(nn.Module):
    def __init__(self, no_of_blocks:int, no_of_heads:int, edim:int, inter_ff:int, dropout:float):
        super().__init__()
        self.blocks = nn.ModuleList([
                        EncoderBlock(no_of_heads, edim, inter_ff, dropout)
                        for _ in range(no_of_blocks)
                    ])
    
    def forward(self, input, input_mask):
        output = input
        for block in self.blocks:
            output = block(output,input_mask)
        return output 

class DecoderBlock(nn.Module):
    def __init__(self, no_of_heads:int, edim:int,
                 inter_f:int, dropout:float):
        super().__init__()
        self.heads = no_of_heads
        self.edim = edim
        self.MHA = MultiHeadAttention(no_of_heads,edim,dropout)
        self.MHCA = MultiHeadAttention(no_of_heads,edim,dropout)
        self.MLP = MLP(edim,inter_f,dropout)
        self.layerNorm1 = nn.LayerNorm(edim)
        self.layerNorm2 = nn.LayerNorm(edim)
        self.layerNorm3 = nn.LayerNorm(edim)
        
    def forward(self,decoder_input,decoder_mask,encoder_output,encoder_mask):
        # input shape (b,context lenght, embed_dim)
        # Masked Multi Head Attention (Input will be masked in the dataset only for this implementation)
        output = self.MHA(decoder_input,decoder_input,decoder_input,decoder_mask)
        # Add and Norm
        output += decoder_input
        output = self.layerNorm1(output)
        
        store = output
        
        # Cross Multi Head Attention (query - decoder , key,value- encoded_features)
        output = self.MHCA(output,encoder_output,encoder_output,encoder_mask)
        # Add and Norm
        output += store
        output = self.layerNorm2(output)
        
        store = output
        
        # Feed Forward 
        output = self.MLP(output)
        # Add and Norm
        output += store
        output = self.layerNorm3(output)
            
        return output
    
class Decoder(nn.Module):
    def __init__(self, no_of_blocks:int, no_of_heads:int, edim:int,
                 inter_f:int, dropout:float):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(no_of_heads, edim, inter_f, dropout)
            for _ in range(no_of_blocks)
        ])
    
    def forward(self, decoder_input, decoder_mask, encoder_output, encoder_mask):
        output = decoder_input
        for block in self.blocks:
            output = block(output, decoder_mask, encoder_output, encoder_mask)
        return output
        
class Transformer(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, no_of_stacks:int, no_of_heads:int, edim:int,
                 intermediate_features:int, pad_idx, dropout:float):
        super().__init__()
        self.encoder = Encoder(no_of_stacks,no_of_heads,edim,
                               intermediate_features,dropout)
        self.decoder = Decoder(no_of_stacks,no_of_heads,edim,
                                intermediate_features,dropout)
        
        self.vocab = nn.Embedding(vocab_size,edim,padding_idx=pad_idx)
        self.positional_embedding = PositionalEncodings(edim,context_length,dropout)
        self.projection = nn.Linear(edim,vocab_size)
        self.edim = edim
        
    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # assuming Input of shape (Batch ,Context lenght) 
        # Converting INput to (Batch, COntext_lenght,edim) by getting the correspoding edim from dictionary
        encoder_input = self.vocab(encoder_input) * math.sqrt(self.edim)
        # Add Positional Embeddings to the input of Encoder
        encoder_input = self.positional_embedding(encoder_input)
        # Encoder
        encoded_features = self.encoder(encoder_input, encoder_mask)  
        
        decoder_input = self.vocab(decoder_input) * math.sqrt(self.edim)
        # Add positional Embeddings to the input of Decoder
        decoder_input = self.positional_embedding(decoder_input)
        # Decoder
        decoded_features = self.decoder(decoder_input, decoder_mask, encoded_features, encoder_mask)
        # Linear layer ( Input - Batch,mtoken, edim) ( Output - Batch, token,vocab size )
        output = self.projection(decoded_features)  
                
        return output

class TransformerSummary(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, no_of_stacks:int, no_of_heads:int, edim:int,
                 intermediate_features:int, pad_idx, dropout:float):
        super().__init__()
        self.encoder = Encoder(no_of_stacks,no_of_heads,edim,
                               intermediate_features,dropout)
        self.decoder = Decoder(no_of_stacks,no_of_heads,edim,
                                intermediate_features,dropout)
        
        self.vocab = nn.Embedding(vocab_size,edim,padding_idx=pad_idx)
        self.positional_embedding = PositionalEncodings(edim,context_length,dropout)
        self.projection = nn.Linear(edim,vocab_size)
        self.edim = edim
        
    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # assuming Input of shape (Batch ,Context lenght) 
        # Converting INput to (Batch, COntext_lenght,edim) by getting the correspoding edim from dictionary
        encoder_input = encoder_input.to(dtype=torch.long)
        encoder_input = self.vocab(encoder_input) * math.sqrt(self.edim)
        # Add Positional Embeddings to the input of Encoder
        encoder_input = self.positional_embedding(encoder_input)
        # Encoder
        encoded_features = self.encoder(encoder_input, encoder_mask)  
        
        decoder_input = decoder_input.to(dtype=torch.long)
        decoder_input = self.vocab(decoder_input) * math.sqrt(self.edim)
        # Add positional Embeddings to the input of Decoder
        decoder_input = self.positional_embedding(decoder_input)
        # Decoder
        decoded_features = self.decoder(decoder_input, decoder_mask, encoded_features, encoder_mask)
        # Linear layer ( Input - Batch,mtoken, edim) ( Output - Batch, token,vocab size )
        output = self.projection(decoded_features)  
        return output

if __name__=='__main__':
    print("Check model by calling it in dataloader file")
    pass