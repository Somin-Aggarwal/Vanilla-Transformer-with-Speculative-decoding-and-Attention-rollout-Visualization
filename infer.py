import pickle
import torch
from torch import nn
from tokenizer import Tokenizer
from dataloader import get_dataloader
from model import Transformer
import argparse

# ——— Utilities ———

def pad_or_truncate(tokens: list, length: int, pad_token: int):
    """Truncate or right-pad token list to exactly `length`."""
    return tokens[:length] + [pad_token] * max(0, length - len(tokens))

def casual_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).to(torch.int)
    return mask == 0

def make_decoder_mask(
    decoder_input: torch.Tensor,  # (1, cur_len)
    pad_token: int,
    max_len: int,
    device: torch.device
):
    """
    Combine padding + causal masks into shape (1,1,cur_len,cur_len).
    """
    # (1, cur_len)
    valid = (decoder_input != pad_token)
    # (1,cur_len,cur_len)
    pad_q = valid.unsqueeze(1)
    pad_k = valid.unsqueeze(2)
    padding_mask = pad_q & pad_k

    # lower‑tri mask (cur_len, cur_len)
    causal = casual_mask(max_len).to(device)

    # (1, cur_len, cur_len) → (1,1,cur_len,cur_len)
    return (padding_mask & causal).unsqueeze(1)

def prepare_inputs(
    sentence: str,
    tokenizer: Tokenizer,
    merge_dict: dict,
    context_length: int,
    sos_token: int,
    pad_token: int,
    eos_token: int,
    device: torch.device
):
    """
    Returns:
      encoder_input:  (1, src_len)
      encoder_mask:   (1,1,1,src_len)
    """
    # tokenize + pad
    tokens = tokenizer.encode_infer(sentence, merge_dict, return_single_list=True)
    tokens = [sos_token] + tokens + [eos_token]
    enc = pad_or_truncate(tokens, context_length, pad_token)
    enc_tensor = torch.tensor(enc, dtype=torch.long, device=device).unsqueeze(0)

    # encoder mask: (1,1,1,src_len)
    enc_valid = (enc_tensor != pad_token)
    enc_mask = enc_valid.unsqueeze(1).unsqueeze(1)

    return enc_tensor, enc_mask

def decode_bytes(tokens: list, vocab: dict) -> str:
    raw = b""
    for idx in tokens:
        if idx == 254:
            break
        raw+=vocab[idx]
    # raw = b"".join(vocab[idx] for idx in tokens)
    return raw.decode("utf-8", errors="replace")


# ——— Main Inference ———

def main(args):
    # config
    WEIGHTS = "weights/best_model.pt"
    VOCAB_PKL = "data_files/vocab.pkl"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONTEXT_LEN = 300
    SOS, EOS, PAD = 0, 254, 255
    MAX_GEN = CONTEXT_LEN

    # model
    model = Transformer(
        vocab_size=8192,
        context_length=CONTEXT_LEN,
        no_of_stacks=6,
        no_of_heads=8,
        edim=512,
        intermediate_features=2048,
        pad_idx=PAD,
        dropout=0.1
    ).to(DEVICE)
    ckpt = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # tokenizer + vocab
    with open(VOCAB_PKL, "rb") as f:
        vocab_bytes, merge_dict = pickle.load(f)
    tokenizer = Tokenizer()
    
    # autoregressive decode
    generated = [SOS]

    # prepare encoder inputs
    input_sentence = args.input_sentence
    enc_in, enc_mask = prepare_inputs(
        input_sentence, tokenizer, merge_dict,
        context_length=CONTEXT_LEN,
        sos_token=SOS, eos_token=EOS, pad_token=PAD,
        device=DEVICE
    )

    for _ in range(MAX_GEN):
        # build decoder_input & mask
        dec_in = torch.tensor(
            pad_or_truncate(generated, CONTEXT_LEN, PAD),
            dtype=torch.long, device=DEVICE
        ).unsqueeze(0)  # (1, CONTEXT_LEN)
        dec_mask = make_decoder_mask(dec_in, PAD, CONTEXT_LEN, DEVICE)
        
        # forward + greedy pick last position
        with torch.no_grad():
            logits = model(enc_in, dec_in, enc_mask, dec_mask)
            probs = torch.softmax(logits,dim=-1)
            last_idx = len(generated) - 1
            last_logits = probs[:, last_idx, :]       # (1, vocab_size)
            next_token = int(last_logits.argmax(dim=-1).item())

        generated.append(next_token)
        if next_token == EOS:
            break

    # convert to text (drop SOS)
    output_text = decode_bytes(generated[1:], vocab_bytes)
    print("Output text:", output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sentence",type=str,required=True)
    args = parser.parse_args()
    main(args)
