from model import BERT
from utilities import *
import yaml
import json


import torch
import torch.nn as nn



import re




def load_yaml(file):
    with open(file, 'r') as file:
        config = yaml.safe_load(file)

    return config


def load_json(file):
    with open(file, 'r') as file:
        textlines = json.load(file)
    return '\n'.join(textlines)






config = load_yaml('config/config.yaml')


d_model = config['d_model']
n_heads = config['n_heads']
n_segments = config['n_segments']
max_len = config['max_len']
n_layers = config['n_layers']
d_k = config['d_k']
d_v = config['d_v']
d_ff = config['d_ff']
max_pred = config['max_pred']
batch_size = config['batch_size']




text = load_json('data/conversation.json')






sentences = re.sub('[.,!?\\-]', '', text.lower()).split('\n')
word_list = list(set(' '.join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

for i, w in enumerate(word_list):
    word_dict[w] = i + 4

number_dict = {i:w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)


token_list = list()

for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)







bert = BERT(d_model, vocab_size, n_heads, n_segments, max_len, n_layers, d_k, d_v, d_ff)






test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "The pen is mightier than the sword.",
]

test_token_list = []
for sentence in test_sentences:
    arr = [word_dict.get(s, word_dict['[MASK]']) for s in sentence.lower().split()]
    test_token_list.append(arr)

tokens_a, tokens_b, _, _ = select_random_sentence_pair(test_sentences, test_token_list)

input_ids, segment_ids = construct_input_segment_ids(tokens_a, tokens_b, word_dict)

masked_tokens, masked_pos, input_ids = mask_tokens(
    input_ids, word_dict, max_pred, vocab_size, number_dict
)

input_ids, segment_ids, masked_tokens, masked_pos = pad_sequences(
    input_ids, segment_ids, masked_tokens, masked_pos, max_len, max_pred
)

input_ids_tensor = torch.LongTensor([input_ids])
segment_ids_tensor = torch.LongTensor([segment_ids])
masked_pos_tensor = torch.LongTensor([masked_pos])

logits_lm, logits_clsf = bert(input_ids=input_ids_tensor, segment_ids=segment_ids_tensor, masked_pos=masked_pos_tensor)

predicted_tokens = logits_lm.data.max(2)[1].numpy()[0]
predicted_masked_tokens = [number_dict[pos] for pos in predicted_tokens if pos != 0]

predicted_isNext = logits_clsf.data.max(1)[1].numpy()[0]

print("Original Sentences:")
print("Sentence A:", " ".join([number_dict[token] for token in tokens_a]))
print("Sentence B:", " ".join([number_dict[token] for token in tokens_b]))
print("\nPredicted Masked Tokens:", predicted_masked_tokens)
print("Predicted isNext:", "True" if predicted_isNext else "False")


