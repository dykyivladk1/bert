from random import *




def get_attn_pad_mask(query_seq, key_seq):
    batch_size, query_len = query_seq.size()
    batch_size, key_len = key_seq.size()
    pad_mask = key_seq.data.eq(0).unsqueeze(1)
    
    pad_mask_expanded = pad_mask.expand(batch_size, query_len, key_len)
    
    return pad_mask_expanded

def select_random_sentence_pair(sentences, token_list):
    tokens_a_index = randrange(len(sentences))
    tokens_b_index = randrange(len(sentences))
    tokens_a = token_list[tokens_a_index]
    tokens_b = token_list[tokens_b_index]
    return tokens_a, tokens_b, tokens_a_index, tokens_b_index




def construct_input_segment_ids(tokens_a, tokens_b, word_dict):
    input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
    segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
    # segment ids represents which tokens belong to the first 
    # sentence and which belong to the second sentence
    # 0 first sentence 1 second sentence
    return input_ids, segment_ids




def mask_tokens(input_ids, word_dict, max_pred, vocab_size, number_dict):
    n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
    cand_makes_pos = [i for i, token in enumerate(input_ids) if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
    shuffle(cand_makes_pos)

    masked_tokens, masked_pos = [], []
    for pos in cand_makes_pos[:n_pred]:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        if random() < 0.8:
            input_ids[pos] = word_dict['[MASK]']
        elif random() < 0.5:
            index = randint(0, vocab_size - 1)
            input_ids[pos] = word_dict[number_dict[index]]

    return masked_tokens, masked_pos, input_ids



def pad_sequences(input_ids, segment_ids, masked_tokens, masked_positions, max_len, max_predictions):
    padding_length = max_len - len(input_ids)
    
    # pad input_ids and segment_ids to max_len
    input_ids.extend([0] * padding_length)
    segment_ids.extend([0] * padding_length)

    # pad masked_tokens and masked_positions if max_predictions is greater
    if max_predictions > len(masked_tokens):
        padding_length = max_predictions - len(masked_tokens)
        masked_tokens.extend([0] * padding_length)
        masked_positions.extend([0] * padding_length)

    return input_ids, segment_ids, masked_tokens, masked_positions



def is_positive_example(tokens_a_index, tokens_b_index):
    return tokens_a_index + 1 == tokens_b_index





def make_batch(sentences, token_list, word_dict, number_dict, batch_size, max_len,
               max_pred, vocab_size):
    positive = negative = 0
    batch = []

    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a, tokens_b, tokens_a_index, tokens_b_index = \
        select_random_sentence_pair(sentences, token_list)

        input_ids, segment_ids = construct_input_segment_ids(tokens_a, tokens_b, word_dict)

        masked_tokens, masked_pos, input_ids = mask_tokens(
            input_ids, word_dict, max_pred, vocab_size, number_dict
        )

        input_ids, segment_ids, masked_tokens, masked_pos = pad_sequences(
            input_ids, segment_ids, masked_tokens, masked_pos, max_len, max_pred
        )

        if is_positive_example(tokens_a_index, tokens_b_index) and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif not is_positive_example(tokens_a_index, tokens_b_index) and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    
    return batch
