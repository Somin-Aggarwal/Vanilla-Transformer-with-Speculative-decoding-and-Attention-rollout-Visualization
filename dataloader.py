import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TranslationDataset(Dataset):
    def __init__(self, encoder_file, decoder_file, context_len=300, pad_token=255, vocab_size=8192):
        self.context_len = context_len
        self.pad_token = pad_token

        with open(encoder_file, "r") as f:
            self.encoder_lines = [list(map(int, line.strip().split())) for line in tqdm(f,desc="Loading Encoder Data")]

        with open(decoder_file, "r") as f:
            decoder_lines = f.readlines()

        self.vocab_size = vocab_size
        self.decoder_inputs = []
        self.targets = []

        for line in tqdm(decoder_lines, total=len(decoder_lines),desc="Loading Decoder Data"):
            tokens = list(map(int, line.strip().split()))
            self.decoder_inputs.append(tokens[:-1])
            self.targets.append(tokens[-1])  # last token is the label

        assert len(self.encoder_lines) == len(self.decoder_inputs), "Mismatch in line counts"
        
    def __len__(self):
        return len(self.encoder_lines)

    def pad_or_truncate(self, tokens):
        if len(tokens) > self.context_len:
            return tokens[:self.context_len]
        return tokens + [self.pad_token] * (self.context_len - len(tokens))

    def __getitem__(self, idx):
        encoder = self.pad_or_truncate(self.encoder_lines[idx])
        decoder_line = self.decoder_inputs[idx]
        decoder = self.pad_or_truncate(decoder_line)

        target_index = self.targets[idx]

        return {
            "encoder_input": torch.tensor(encoder, dtype=torch.int),
            "decoder_input": torch.tensor(decoder, dtype=torch.int),
            "target": torch.tensor(target_index, dtype=torch.long),
            "context_index" : len(decoder_line)-1,
        }

def get_dataloader(encoder_path, decoder_path, batch_size=2, shuffle=True, seed=None):
    dataset = TranslationDataset(encoder_path, decoder_path)

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
    
if __name__=="__main__":
    encoder_file = "encodings_val.txt"
    decoder_file = "decodings_val.txt"
    dataloader = get_dataloader(encoder_file,decoder_file, batch_size=1, shuffle=False)
    print(len(dataloader))
    for i,data in enumerate(dataloader):
        # print(data)
        if i== 10:
            break