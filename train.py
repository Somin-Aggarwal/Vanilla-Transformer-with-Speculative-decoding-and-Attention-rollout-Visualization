import torch
import torch.nn as nn
from torch.optim import Adam,AdamW
import os 
from dataloader import get_dataloader
from model import Transformer
import argparse
from tqdm import tqdm
import logging
import torch.optim.lr_scheduler as lr_scheduler
from argparse import Namespace
import sys

logger = logging.getLogger(__name__)

def save_checkpoint(filename, model_state_dict, optimizer_state_dict,
                    scheduler_state_dict, epoch, iteration, loss, args, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'loss': loss,
        'training_config': vars(args)
    }
    path = os.path.join(weights_dir, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    logging.info(f"Saved checkpoint {filename} at epoch {epoch}, iter {iteration}, loss {loss}")


def train(args, config=None, resume=False):
    
    logging.basicConfig(
    filename=args.log_file,
    filemode='a',  # Append mode; creates file if it doesn't exist
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)   
    logger.info("\n\n\nSTARTED TRAINING\n")
    print(vars(args))
    logger.info(f"{vars(args)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    import random
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(args.weights_dir, exist_ok=True)
    
    train_loader = get_dataloader(
        input_language_path=args.train_input_language_path,
        output_language_path=args.train_output_language_path,
        vocab_merge_file=args.vocab_merge_file,
        context_length=args.context_length,
        pad_token=args.padding_index,
        already_tokenized=args.already_tokenized,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.random_seed
    )

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

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        no_of_stacks=args.no_of_blocks,
        no_of_heads=args.no_of_attn_heads,
        edim=args.edim,
        intermediate_features=args.mlp_intermediate_features,
        pad_idx=args.padding_index,
        dropout=args.dropout
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=args.padding_index)
    optimizer = AdamW(params=model.parameters(),lr=args.learning_rate, betas=(args.beta1,args.beta2), eps=args.eps)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    
    start_epoch = 0
    skip_iteration = -1
    if os.path.exists(f"{args.weights_dir}/best_model.pt"):
        best_val_loss = torch.load(f"{args.weights_dir}/best_model.pt")['loss']
    else:
        best_val_loss = 1e10
    
    # Resume from checkpoint
    if resume and config is not None:
        # Load saved states
        start_epoch = config['epoch']
        skip_iteration = config['iteration']
        model.load_state_dict(config['model_state_dict'])
        optimizer.load_state_dict(config['optimizer_state_dict'])
        if 'scheduler_state_dict' in config:
            scheduler.load_state_dict(config['scheduler_state_dict'])
        print(f"Resuming from epoch {start_epoch}, iteration {skip_iteration}")
        logging.info(f"Resumed: epoch {start_epoch}, iter {skip_iteration}, best_val_loss {best_val_loss}")
    
    epochs = args.epochs
    total_batches = len(train_loader)
    # if save_every_n_iteration_percentage = 5 then 100/5 = 20 files will be stored for every iteration
    save_every = max(1, total_batches * args.save_every_n_iteration_percentage // 100)
    validate_every = max(1, total_batches * args.validation_every_n_iteration_percentage // 100)
    
    print(f"Total batches: {total_batches}, save every {save_every}, validate every {validate_every}")
    logging.info(f"Total batches: {total_batches}, save_every: {save_every}, validate_every: {validate_every}")
    
    for current_epoch in tqdm(range(start_epoch,epochs),desc="Epochs"):
        
        epoch_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {current_epoch+1}", leave=False)
        
        for current_iteration, batch in enumerate(train_iterator):
            # skip already processed iterations
            if current_epoch == start_epoch and current_iteration <= skip_iteration:
                continue

            enc_input = batch['encoder_input'].to(args.device)
            dec_input = batch['decoder_input'].to(args.device)
            target = batch['target'].to(args.device)
            enc_mask = batch['encoder_mask'].to(args.device)
            dec_mask = batch['decoder_mask'].to(args.device)
            
            optimizer.zero_grad()
            
            output = model(enc_input, dec_input, enc_mask, dec_mask)

            output = output.reshape(-1, args.vocab_size)
            target = target.reshape(-1)
            loss = criterion(output, target)   
                        
            loss.backward()
            optimizer.step()
            
            loss_value = loss.item() 
            epoch_loss += loss_value

            # Display iteration loss on tqdm bar
            train_iterator.set_postfix(iter_loss=loss_value)
            logger.info(f"Epoch : {current_epoch} | Iteration : {current_iteration} | Iteration Loss : {loss_value} | Running Epoch Loss : {epoch_loss}")
            
            # Save intermediate checkpoint
            if (current_iteration + 1) % save_every == 0:
                save_checkpoint(
                    f"checkpoint_epoch{current_epoch}_iter{current_iteration}.pt",
                    model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                    current_epoch, current_iteration, loss_value,
                    args, args.weights_dir
                )
                prev_file = f"{args.weights_dir}/checkpoint_epoch{current_epoch-1}_iter{current_iteration}.pt"
                if os.path.isfile(prev_file):
                    os.remove(prev_file)

            # Validation
            if (current_iteration + 1) % validate_every == 0 or current_iteration == total_batches - 1:
                model.eval()
                val_loss = 0.0
                for i, vbatch in enumerate(val_loader):
                    enc_v = vbatch['encoder_input'].to(args.device)
                    dec_v = vbatch['decoder_input'].to(args.device)
                    tgt_v = vbatch['target'].to(args.device)
                    enc_mask_v = vbatch['encoder_mask'].to(args.device)
                    dec_mask_v = vbatch['decoder_mask'].to(args.device)
                    with torch.no_grad():
                        out_v = model(enc_v, dec_v, enc_mask_v, dec_mask_v)
                        out_v = out_v.reshape(-1, args.vocab_size)
                        tgt_v = tgt_v.reshape(-1)
                        l_v = criterion(out_v, tgt_v)   
                    val_loss += l_v.item()
                val_loss /= (i + 1)
                print(f"Validation Loss: {val_loss}")
                logging.info(f"Validation Loss {val_loss} at epoch {current_epoch}, iter {current_iteration}")
                
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        "best_model.pt",
                        model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                        current_epoch, current_iteration, val_loss,
                        args, args.weights_dir
                    )
                    print(f"Saving model as BEST MODEL")
                model.train()

        # Epoch-end save
        if (current_epoch + 1) % args.save_every_n_epochs == 0 or current_epoch == epochs -1 :
            avg_loss = epoch_loss / (total_batches - skip_iteration if current_epoch == start_epoch else total_batches)
            save_checkpoint(
                f"epoch{current_epoch+1}_end.pt",
                model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                current_epoch, current_iteration, avg_loss,
                args, args.weights_dir
            )
        # after first resume epoch, reset skip_iteration
        skip_iteration = -1      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data Paths
    parser.add_argument("--train_input_language_path", type=str, default="data_files/encodings_train.txt")
    parser.add_argument("--train_output_language_path", type=str, default="data_files/decodings_train.txt")
    parser.add_argument("--val_input_language_path", type=str, default="data_files/encodings_val_all.txt")
    parser.add_argument("--val_output_language_path", type=str, default="data_files/decodings_val_all.txt")
    parser.add_argument("--already_tokenized",type=bool,default=True)
    
    # Transformer Hyperparameters (Default = Vanilla Transformer)
    parser.add_argument("--no_of_blocks", type=int, default=6)
    parser.add_argument("--no_of_attn_heads", type=int, default=8)
    parser.add_argument("--edim", type=int, default=512)
    parser.add_argument("--mlp_intermediate_features", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--padding_index", type=int, default=255)
    parser.add_argument("--context_length", type=int, default=300)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training Args
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--save_every_n_iteration_percentage", type=int, default=50)
    parser.add_argument("--validation_every_n_iteration_percentage", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--log_file", type=str, default="training_logs.log")
    parser.add_argument("--vocab_merge_file", type=str, default="data_files/vocab.pkl")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Enter the path to resume training
    weights_path = None
    args.resume_from = weights_path

    # While resuming, all the args are overwritten by the args in the pt file 
    checkpoint_config = None
    resume = False
    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint_config = torch.load(args.resume_from)
        args = Namespace(**checkpoint_config["training_config"])
        args.resume_from = weights_path
        resume = True
    
    train(args, checkpoint_config, resume)
