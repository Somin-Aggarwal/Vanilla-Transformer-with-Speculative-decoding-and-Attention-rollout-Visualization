import pickle

import torch
import torch.nn as nn

from training_tokenier import Tokenizer
from model import Transformer


def pad_or_truncate(
    tokens: list, 
    context_length: int, 
    pad_token: int
) -> list:
    """
    Ensure the token list is exactly context_length:
    - Truncate if longer.
    - Pad with pad_token if shorter.
    """
    if len(tokens) > context_length:
        return tokens[:context_length]
    return tokens + [pad_token] * (context_length - len(tokens))


def decode(tokens_list: list, vocab: dict) -> str:
    """
    Reconstruct a UTF-8 string from a list of token indices.
    Unknown bytes are replaced.
    """
    raw = b"".join(vocab[idx] for idx in tokens_list)
    return raw.decode("utf-8", errors="replace")


def main():
    # --------- Configuration ---------
    weights_path = "checkpoint_epoch2_iter33596.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Model Setup -----------
    model = Transformer(
        vocab_size=8192,
        no_of_stacks=3,
        no_of_heads=8,
        edim=512,
        in_features=512,
        intermediate_features=2048,
        out_features=512,
        context_length=300,
        pad_idx=255,
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --------- Tokenizer & Vocab -----
    tokenizer = Tokenizer()
    with open("vocab.pkl", "rb") as f:
        vocab, merge_dict = pickle.load(f)

    # ---- Input Sentence & Encoding ----
    input_sentence =  "I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night."

    encoder_tokens = tokenizer.encode_infer(
        text=input_sentence,
        merge_dictionary=merge_dict,
        return_single_list=True,
    )
    encoder_tokens = pad_or_truncate(encoder_tokens, context_length=300, pad_token=255)
    encoder_input = torch.tensor(encoder_tokens, dtype=torch.long)
    encoder_input = encoder_input.unsqueeze(0).to(device)

    # --------- Autoregressive Decoding ---------
    decoded_tokens: list[int] = []
    pad_token = 255
    eos_token = 254
    max_len = 300

    # Start decoding with the EOS token (or any start symbol)
    next_token = eos_token
    for _ in range(max_len):
        decoded_tokens.append(next_token)
        decoder_tokens = pad_or_truncate(decoded_tokens, context_length=300, pad_token=pad_token)
        decoder_input = torch.tensor(decoder_tokens, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(encoder_input, decoder_input)
            probs = torch.softmax(logits, dim=-1)
            next_token = int(torch.argmax(probs, dim=-1).item())

        if next_token == eos_token:
            break

    # ---- Decode and Print ----
    # Remove the initial EOS token from output if you added it
    output_text = decode(decoded_tokens[1:], vocab)
    print("Generated tokens:", decoded_tokens)
    print("Output text:", output_text)


if __name__ == "__main__":
    main()
