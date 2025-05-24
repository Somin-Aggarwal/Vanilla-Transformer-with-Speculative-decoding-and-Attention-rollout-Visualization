import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizer import Tokenizer
import pickle
import matplotlib.pyplot as plt
from model import TransformerSummary, Transformer

# USing this the diagonal and lower triangular matrices are true rest false
def casual_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).to(torch.int)
    return mask == 0

class TranslationDataset(Dataset):
    def __init__(
        self,
        input_language_file: str,
        output_language_file: str,
        vocab_merge_file: str,
        context_len: int = 300,
        pad_token: int = 255,
        already_tokenized: bool = False,
    ):
        super().__init__()
        self.context_len = context_len
        self.pad_token = pad_token
        self.sos_token = 0
        self.eos_token = 254
        self.already_tokenized = already_tokenized

        with open(input_language_file, "r") as f:
            self.english_lines = f.readlines()

        with open(output_language_file, "r") as f:
            self.german_lines = f.readlines()

        with open(vocab_merge_file, "rb") as f:
            self.vocab, self.merge_dict = pickle.load(f)

        assert len(self.english_lines) == len(self.german_lines), \
            f"Mismatch in data lengths: {len(self.english_lines)} vs {len(self.german_lines)}"

        if not self.already_tokenized:
            self.tokenizer = Tokenizer()

    def casual_mask(self, size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).to(torch.int)
        return mask == 0

    def __len__(self):
        return len(self.english_lines)

    def pad_or_truncate(self, tokens):
        if len(tokens) > self.context_len:
            return tokens[: self.context_len]
        return tokens + [self.pad_token] * (self.context_len - len(tokens))

    def __getitem__(self, idx: int):
        english_line = self.english_lines[idx].strip()
        german_line = self.german_lines[idx].strip()

        if self.already_tokenized:
            enc_tokens = list(map(int, english_line.split()))
            dec_tokens = list(map(int, german_line.split()))
        else:
            enc_tokens = self.tokenizer.encode_infer(
                english_line, self.merge_dict, return_single_list=True
            )
            dec_tokens = self.tokenizer.encode_infer(
                german_line, self.merge_dict, return_single_list=True
            )
            enc_tokens = [self.sos_token] + enc_tokens + [self.eos_token]
            dec_tokens = [self.sos_token] + dec_tokens + [self.eos_token]


        decoder_input = dec_tokens[:-1]       # [SOS] + sentence tokens
        decoder_output = dec_tokens[1:]       # sentence tokens + [EOS]
        
        enc_padded = self.pad_or_truncate(enc_tokens)
        dec_in_padded = self.pad_or_truncate(decoder_input)
        dec_out_padded = self.pad_or_truncate(decoder_output)

        encoder_input = torch.tensor(enc_padded, dtype=torch.long)
        decoder_input = torch.tensor(dec_in_padded, dtype=torch.long)
        decoder_output = torch.tensor(dec_out_padded, dtype=torch.long)

        encoder_mask = (encoder_input != self.pad_token)
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)

        decoder_mask = (decoder_input != self.pad_token)
        pad_q = decoder_mask.unsqueeze(1)
        pad_k = decoder_mask.unsqueeze(0)
        padding_mask = pad_q & pad_k
        casual_mask = self.casual_mask(self.context_len)
        decoder_mask = (padding_mask & casual_mask)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "target": decoder_output,
            "english_text": english_line,
            "german_text": german_line,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask
        }

def get_dataloader(input_language_path, output_language_path, vocab_merge_file, context_length, pad_token, already_tokenized,batch_size=2, shuffle=True, seed=None):
    dataset = TranslationDataset(input_language_path, output_language_path, vocab_merge_file, context_length, pad_token, already_tokenized)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=(lambda worker_id: torch.manual_seed(seed + worker_id)) if seed is not None else None,
        generator=generator
    )
    
if __name__ == "__main__":
    
    from torchsummary import summary
    
    encoder_file = "data_files/encodings_train.txt"
    decoder_file = "data_files/decodings_train.txt"
    vocab_merge_file = "data_files/vocab.pkl"
    context_length = 300
    pad_token = 255
    device = 'cuda'

    dataloader = get_dataloader(
        encoder_file, decoder_file, vocab_merge_file,
        context_length, pad_token, already_tokenized=False,
        batch_size=1, shuffle=False
    )
    print(len(dataloader))

    no_of_stacks = 6
    no_of_heads = 8
    edim = 512
    intermediate_features = 2048
    vocab_size = 8192
    pad_idx = 255
    dropout = 0.1

    transformer = Transformer(
        vocab_size, context_length, no_of_stacks, no_of_heads,
        edim, intermediate_features, pad_idx, dropout
    ).to(device)

    for i, data in enumerate(dataloader):
        encoder_input = data["encoder_input"].to(device)
        decoder_input = data["decoder_input"].to(device)
        yarget = data["target"].to(device)  # assuming "yarget" is a typo for "target"
        english_text = data["english_text"]  # Keep on CPU since it’s string
        german_text = data["german_text"]    # Keep on CPU since it’s string
        encoder_mask = data["encoder_mask"].to(device)
        decoder_mask = data["decoder_mask"].to(device)        
        
        # print(f"Dataloader encoder input : {encoder_input}")
        # encoder_input = transformer.vocab(encoder_input)
        # encoder_input = transformer.positional_embedding(encoder_input)
        # encoded_features = transformer.encoder(encoder_input,encoder_mask)
        # print(torch.isnan(encoded_features).any())
        # decoder_input = transformer.vocab(decoder_input)
        # decoder_input = transformer.positional_embedding(decoder_input)
        # print(torch.isnan(decoder_input).any())
        # output = transformer.decoder.blocks[0].MHA(decoder_input,decoder_input,decoder_input,decoder_mask)
        # print(f"Encoder_input : {encoder_input[0,:,0]}")
        # print(f"Decoder input : {decoder_input[0,:,0]}")
        # print(f"Output : {output}")
        # print(torch.isnan(output).any())
        
        # output = transformer(encoder_input, decoder_input, encoder_mask, decoder_mask)
        
        break 

        