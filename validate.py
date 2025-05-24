import torch
import torch.nn as nn
import argparse
import os
from dataloader import get_dataloader
from model import Transformer
from tqdm import tqdm
import pickle

def decode(tokens, vocab):
    sentence = b""
    for idx in tokens:
        if idx==254:
            break
        sentence += vocab[idx]
    return sentence.decode("utf-8", errors="replace")

def validate(args):

    with open(args.vocab_merge_file,"rb") as file:
        vocab, merge_dict = pickle.load(file)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    assert os.path.isfile(args.checkpoint_path), f"Checkpoint not found: {args.checkpoint_path}"
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint.get('training_config', {})

    # Build model
    model = Transformer(
        vocab_size=config.get('vocab_size', args.vocab_size),
        context_length=config.get('context_length', args.context_length),
        no_of_stacks=config.get('no_of_blocks', args.no_of_blocks),
        no_of_heads=config.get('no_of_attn_heads', args.no_of_attn_heads),
        edim=config.get('edim', args.edim),
        intermediate_features=config.get('mlp_intermediate_features', args.mlp_intermediate_features),
        pad_idx=config.get('padding_index', args.padding_index),
        dropout=config.get('dropout', args.dropout)
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare validation data loader
    val_loader = get_dataloader(
        input_language_path=args.val_input_language_path,
        output_language_path=args.val_output_language_path,
        vocab_merge_file=args.vocab_merge_file,
        context_length=args.context_length,
        pad_token=args.padding_index,
        already_tokenized=args.already_tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.random_seed
    )

    # Loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=args.padding_index)

    # Validation loop
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(val_loader,total=len(val_loader)):
            enc_input = batch['encoder_input'].to(device)
            dec_input = batch['decoder_input'].to(device)
            target = batch['target'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            dec_mask = batch['decoder_mask'].to(device)

            output_model = model(enc_input, dec_input, enc_mask, dec_mask)

            outputs = torch.argmax(output_model,dim=-1)
            for i in range(len(outputs)):
                print("\n")
                print(batch['english_text'][i])
                print(batch['german_text'][i])
                output = outputs[i]
                output = output.cpu().tolist()
                sentence = decode(output,vocab)
                print(sentence)

            output_model = output_model.view(-1, output_model.size(-1))
            target = target.view(-1)

            loss = criterion(output_model, target)
            total_loss += loss.item() * target.ne(args.padding_index).sum().item()
            total_tokens += target.ne(args.padding_index).sum().item()

    avg_loss = total_loss / total_tokens
    print(f"Validation results on {len(val_loader.dataset)} samples:")
    print(f"  Average per-token loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation-only script for Transformer model")

    parser.add_argument('--checkpoint_path', type=str, 
                        help='Path to the saved .pt checkpoint', default="weights/best_model.pt")
    parser.add_argument('--val_input_language_path', type=str, 
                        help='Path to the validation encoder inputs',default="data_files/english_val.txt")
    parser.add_argument('--val_output_language_path', type=str, 
                        help='Path to the validation decoder inputs',default="data_files/german_val.txt")
    parser.add_argument('--vocab_merge_file', type=str, 
                        help='Path to the vocab merge .pkl file',default="data_files/vocab.pkl")
    parser.add_argument('--context_length', type=int, default=300,
                        help='Maximum sequence length')
    parser.add_argument('--padding_index', type=int, default=255,
                        help='Padding token index')
    parser.add_argument('--already_tokenized', type=bool, default=False,
                        help='Whether data is already tokenized')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to run validation on')
    parser.add_argument('--no_of_blocks', type=int, default=6,
                        help='Number of transformer blocks (fallback if not in checkpoint)')
    parser.add_argument('--no_of_attn_heads', type=int, default=8,
                        help='Number of attention heads (fallback)')
    parser.add_argument('--edim', type=int, default=512,
                        help='Embedding dimension (fallback)')
    parser.add_argument('--mlp_intermediate_features', type=int, default=2048,
                        help='MLP intermediate features size (fallback)')
    parser.add_argument('--vocab_size', type=int, default=8192,
                        help='Vocabulary size (fallback)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (fallback)')

    args = parser.parse_args()
    validate(args)
