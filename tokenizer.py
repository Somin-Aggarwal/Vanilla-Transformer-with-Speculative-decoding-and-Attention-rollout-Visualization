import regex as re
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import partial
from collections import Counter
from line_profiler import profile
    
class Tokenizer():
    def __init__(self, language_file_input=None, language_file_target=None, vocab_size=8192
                 ,save_path="tokens_info.pkl"):
        self.regex = re.compile(r"""
                    '(?i:[sdmt]|ll|ve|re)|          # Contractions
                    (?:\p{P}|\p{S})+|                # Punctuation/Symbols (new rule)
                    [^\r\n\p{L}\p{N}]?+\p{L}++|      # Words with optional non-alphanumeric prefix
                    \p{N}{1,3}+|                     # Numbers
                    \s++$|                           # Trailing whitespace
                    \s*[\r\n]|                       # Newlines
                    \s+(?!\S)|                       # Whitespace gaps
                    \s                               # Any remaining whitespace
                    """, flags=re.X | re.UNICODE)
        if language_file_input != None:
            self.language_file_input = language_file_input
            self.language_file_target = language_file_target
            self.chunks = self.process_data()
            self.chunk_tokens = self.get_tokens(self.chunks)
            self.vocab , self.merge_dictionary = self.encode(vocab_size,save_path,self.chunk_tokens)

    def process_data(self):
        with open(self.language_file_input, 'r') as input_data:
            input_data = input_data.readlines()
        with open(self.language_file_target, 'r') as target_data:
            target_data = target_data.readlines()
            
        final_data = input_data + target_data
        del input_data, target_data
        
        final_data = "".join(line.strip() for line in final_data)
        
        chunks = re.findall(self.regex,final_data)
        return chunks

    def get_tokens(self, *args):
        tokens = []
        if len(args) == 0:
            chunks = self.chunks
        else:
            chunks = args[0]
        
        for text in chunks:
            tokens_bytes = text.encode("utf-8")
            tokens_int = list(map(int, tokens_bytes))
            tokens.append(tokens_int)
        return tokens

    def get_stats_original(self, *args):
        dictionary = {}
        if len(args) == 0:
            chunk_tokens = self.chunk_tokens
        else:
            chunk_tokens = args[0]
            
        for tokens in chunk_tokens:
            for i in range(0,len(tokens)-1):
                pair = (tokens[i],tokens[i+1])
                if pair in dictionary:
                    dictionary[pair] += 1
                else:
                    dictionary[pair] = 1
        return dictionary 
    
    def get_stats(self, *args):
        chunk_tokens = self.chunk_tokens if not args else args[0]
        return Counter((a, b) for tokens in chunk_tokens 
                   for a, b in zip(tokens, tokens[1:]))
    
    def merge(self,pair,idx,chunk_tokens):
        pair_0, pair_1 = pair
        merged_chunk_tokens = []
        for tokens in chunk_tokens:
            new_tokens = []
            i=0
            n = len(tokens)
            while i < n-1:
                if tokens[i] == pair_0 and tokens[i+1] == pair_1:
                    new_tokens.append(idx)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            if(i < n):
                new_tokens.append(tokens[i])
                
            merged_chunk_tokens.append(new_tokens)
        
        return merged_chunk_tokens

    def encode(self, total_vocab_size, save_path, *args):
        vocab = {idx:bytes([idx]) for idx in range(256)}
        idx=256
        total_merges = total_vocab_size - 256
        idx_info = {}
        
        if len(args) == 0:
            chunk_tokens = self.chunk_tokens
        else:
            chunk_tokens = args[0]
        
        merged_chunk_tokens = chunk_tokens
        for _ in tqdm(range(total_merges), desc="Merging Pairs"):
            all_pairs = self.get_stats(merged_chunk_tokens)
            if not all_pairs:
                print("\n\nBREAKING AS NO MORE PAIRS\n\n")
                break 
            pair = max(all_pairs, key=all_pairs.get)
            idx_info[pair] = idx
            merged_chunk_tokens = self.merge(pair, idx, merged_chunk_tokens)
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            idx += 1
                
        with open(save_path, 'wb') as f:
            pickle.dump((vocab, idx_info), f)
                    
        return vocab, idx_info

    def decode(self, chunk_tokens, vocab):
        all_bytes = []
        for tokens in chunk_tokens:
            all_bytes.extend(vocab[idx] for idx in tokens)
        sentence = b"".join(all_bytes)
        return sentence.decode("utf-8", errors="replace")

    @profile
    def encode_infer(self, text, merge_dictionary, return_single_list = False):
        chunks = re.findall(self.regex, text)
        chunk_tokens = self.get_tokens(chunks)
        
        i=0
        while True:
            all_pairs = self.get_stats(chunk_tokens)
            if len(all_pairs) == 0:
                break
            pair = min(all_pairs, key=lambda p: merge_dictionary.get(p, float("inf")))
            if pair not in merge_dictionary:
                break
            idx = merge_dictionary[pair]
            chunk_tokens = self.merge(pair, idx, chunk_tokens)
            i+=1
                
        if return_single_list:
            single_list = []
            for tokens in chunk_tokens:
                for token in tokens:
                    single_list.append(token)
            return single_list
        
        # print(f"Total {i} Pairs found")
        return chunk_tokens
    
    
if __name__ == '__main__':
    english_file_path = "en-de/english_processed.txt"
    german_file_path = "en-de/german_processed.txt"
    tokenizer = Tokenizer(language_file_input=english_file_path,
                          language_file_target=german_file_path,
                          vocab_size=8000,save_path="main.pkl")