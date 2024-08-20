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




tokens_a, tokens_b, tokens_a_index, tokens_b_index = select_random_sentence_pair(sentences, token_list)




input_ids, segment_ids = construct_input_segment_ids(tokens_a,
                                                     tokens_b,
                                                     word_dict)



masked_tokens, masked_pos, input_ids = mask_tokens(
    input_ids, word_dict, max_pred, vocab_size, number_dict
)


input_ids, segment_ids, masked_tokens, masked_pos = pad_sequences(
    input_ids, segment_ids, masked_tokens, masked_pos, max_len, max_pred
)

batch = make_batch(sentences, token_list, word_dict, number_dict,
                   batch_size, max_len, max_pred, vocab_size)



input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(* batch))


bert = BERT(d_model, vocab_size, n_heads, n_segments, max_len, n_layers, d_k, d_v, d_ff)



bert.forward(input_ids = input_ids, segment_ids = segment_ids, masked_pos = masked_pos)

optimizer = torch.optim.Adam(bert.parameters(), lr = 0.00001)

criterion = nn.CrossEntropyLoss()



for epoch in range(300):
    optimizer.zero_grad()

    logits_lm, logits_clsf = bert(input_ids, segment_ids, masked_pos)
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)

    loss_lm = (loss_lm.float()).mean()
    loss_clsf = criterion(logits_clsf, isNext)
    
    loss = loss_lm + loss_clsf
    if (epoch + 1) % 10 == 0:
        print('epoch: [{}] cost: [{}]'.format(epoch, loss))
    loss.backward()
    optimizer.step()


torch.save(bert.state_dict(), 'weights/bert.pt')