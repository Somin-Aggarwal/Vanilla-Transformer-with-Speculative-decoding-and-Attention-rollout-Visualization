import torch
import torch.nn as nn
from my_utils import MultiHeadAttention,MLP, PositionalEncodings
from torchsummary import summary
import math

class Encoder(nn.Module):
    def __init__(self, no_of_stacks:int, no_of_heads:int, edim:int, inter_ff:int, droupout:float):
        super().__init__()
        self.no_of_stacks = no_of_stacks
        self.heads = no_of_heads
        self.edim = edim
        self.MHA = MultiHeadAttention(no_of_heads,edim,droupout)
        self.MLP = MLP(edim,inter_ff,droupout)
        self.layerNorm = nn.LayerNorm(edim)
        
    def forward(self,encoder_input, encoder_input_mask):
        # Input Shape (Batch, Context Lenght, Embedding_dim)
        assert encoder_input.ndim==3
        
        output = encoder_input
        for i in range(self.no_of_stacks):
            store = output
            
            # Multi Head Attention
            output = self.MHA(output,output,output,encoder_input_mask)
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
                 inter_f,droupout):
        super().__init__()
        self.no_of_stacks = no_of_stacks
        self.heads = no_of_heads
        self.edim = edim
        self.MHA = MultiHeadAttention(no_of_heads,edim,droupout)
        self.MLP = MLP(edim,inter_f,droupout)
        self.layerNorm = nn.LayerNorm(edim)
        
    def forward(self,decoder_input,decoder_mask,encoder_output,encoder_mask):
        '''
        The Decoders input is the output of decoder itself .
        The start token is given as input to the model at the beginning 
        '''
        # input shape (b,context lenght, embed_dim)
        output = decoder_input
        for i in range(self.no_of_stacks):
            store = output
            
            # Masked Multi Head Attention (Input will be masked in the dataset only for this implementation)
            output = self.MHA(output,output,output,decoder_mask)
            # Add and Norm
            output += store
            output = self.layerNorm(output)
            
            store = output
            
            # Cross Multi Head Attention (query - decoder , key,value- encoded_features)
            output = self.MHA(output,encoder_output,encoder_output,encoder_mask)
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
        
        # Softmax (input - (batch , tokens, vocab_size) 
            # output = torch.softmax(output,dim=-1)
        # Return the predicted Token
            # output = torch.argmax(output,dim=-1)   
        
        return output
    
class Transformer_dummy(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, no_of_stacks:int, no_of_heads:int, edim:int,
                 intermediate_features:int, pad_idx, droupout:float):
        super().__init__()
        self.encoder = Encoder(no_of_stacks,no_of_heads,edim,
                               intermediate_features,droupout)
        self.decoder = Decoder(no_of_stacks,no_of_heads,edim,
                      intermediate_features,droupout)
        
        self.vocab = nn.Embedding(vocab_size,edim,padding_idx=pad_idx)
        self.positional_embedding = PositionalEncodings(edim,context_length,droupout)
        self.projection = nn.Linear(edim,vocab_size)
        self.edim = edim
        
    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # assuming Input of shape (Batch ,Context lenght) 
        # Converting INput to (Batch, COntext_lenght,edim) by getting the correspoding edim from dictionary
       
        
        encoder_input = torch.randint(1, vocab_size, (batch_size, context_length), dtype=torch.int64).to(device)
        decoder_input = torch.randint(1, vocab_size, (batch_size, context_length), dtype=torch.int64).to(device)

        # Encoder mask: 1 for non-padding, 0 for padding
        encoder_mask = (encoder_input != pad_idx).unsqueeze(1).unsqueeze(2).int().to(device)  # (B, 1, 1, T)

        # Causal mask + padding mask for decoder
        def generate_causal_mask(size):
            mask = torch.triu(torch.ones((1, size, size), dtype=torch.int), diagonal=1)
            return mask == 0

        decoder_pad_mask = (decoder_input != pad_idx).unsqueeze(1).int().to(device)  # (B, 1, T)
        causal = generate_causal_mask(context_length).to(device)  # (1, T, T)
        decoder_mask = decoder_pad_mask.unsqueeze(1) & causal  # (B, 1, T, T)
        decoder_mask = decoder_mask.to(device)
        print(decoder_mask.shape)


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
        
        # Softmax (input - (batch , tokens, vocab_size) 
            # output = torch.softmax(output,dim=-1)
        # Return the predicted Token
            # output = torch.argmax(output,dim=-1)   
        
        return output
    
if __name__=='__main__':
    no_of_stacks = 6
    no_of_heads = 8
    edim = 512
    intermediate_features = 2048
    device = 'cuda'
    context_length = 300
    vocab_size = 50000
    pad_idx = 255
    droupout = 0.1
        
    dummy_input = torch.rand(size=(3,context_length,edim)).to(device)

    transformer = Transformer_dummy(vocab_size,context_length, no_of_stacks, no_of_heads,edim,
                              intermediate_features, pad_idx, droupout).to(device)
    
    batch_size=1
    encoder_input = torch.randint(1, vocab_size, (batch_size, context_length), dtype=torch.int64).to(device)
    decoder_input = torch.randint(1, vocab_size, (batch_size, context_length), dtype=torch.int64).to(device)

    # Encoder mask: 1 for non-padding, 0 for padding
    encoder_mask = (encoder_input != pad_idx).unsqueeze(1).unsqueeze(2).int().to(device)  # (B, 1, 1, T)

    # Causal mask + padding mask for decoder
    def generate_causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size), dtype=torch.int), diagonal=1)
        return mask == 0

    decoder_pad_mask = (decoder_input != pad_idx).unsqueeze(1).int().to(device)  # (B, 1, T)
    causal = generate_causal_mask(context_length).to(device)  # (1, T, T)
    decoder_mask = decoder_pad_mask.unsqueeze(1) & causal  # (B, 1, T, T)
    decoder_mask = decoder_mask.to(device)
    
    output = transformer(encoder_input, decoder_input, encoder_mask, decoder_mask)
    
    # for name, param in transformer.named_parameters():
    #     print(name, param.grad is not None)

    # Use input_data instead of input_size
    # summary(transformer, input_size=[(context_length, edim), (context_length, edim), (context_length, edim), (context_length, edim)], batch_size=1)    
    # vocab = nn.Embedding(8192,512)
    # a = torch.zeros(size=(1,300),dtype=torch.long)
    # b = vocab(a)
    # print(b.dtype)