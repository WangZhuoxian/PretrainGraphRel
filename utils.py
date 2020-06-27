#!/usr/bin/env/ python
# coding: utf-8
"""
@File: utils.py
@Description: Void
@Copyright: Zhuoxian Wang
@Email: sysu14336205@163.com
@Date: 2020/3/26
"""
import numpy as np
import torch
import json
from collections import defaultdict
import spacy
from tqdm import tqdm
from torch.utils.data import Dataset

from const import *


def read_txt(file_path, directional_relation=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    indexes, sentences, relations = [], [], []
    for l in range(len(lines) // 4):
        index, sentence = lines[l * 4].split('\t')
        index = int(index)
        assert sentence[-1:] == "\n"
        sentence = sentence[:-1].strip("\"").strip()
        relation = lines[l * 4 + 1].strip()
        indexes.append(index)
        sentences.append(sentence)
        relations.append(relation)
    if not directional_relation:
        for r_i in range(len(relations)):
            relations[r_i] = relations[r_i].replace("(e2,e1)", "").replace("(e1,e2)", "")
    return indexes, sentences, relations


def remove_ent_mask(text):
    assert '<e1>' in text and '</e1>' in text and '<e2>' in text and '</e2>' in text
    text = keep_en_tag_wrapped_with_space(text)
    text = text.split()
    a, b, c, d = text.index('<e1>'), text.index('</e1>'), text.index('<e2>'), text.index('</e2>')
    assert a < b < c < d
    text.pop(d)
    text.pop(c)
    text.pop(b)
    text.pop(a)
    b -= 2
    c -= 2
    d -= 4
    ls = list(map(len, text))
    cum_sum = [0]
    cum_sum.extend(ls)
    for l in range(1, len(cum_sum)):
        cum_sum[l] = cum_sum[l - 1] + cum_sum[l]
    chr_a = cum_sum[a]
    chr_c = cum_sum[c]
    chr_b = cum_sum[b + 1] - 1
    chr_d = cum_sum[d + 1] - 1
    return ' '.join(text), ((chr_a, chr_b), (chr_c, chr_d))


def keep_en_tag_wrapped_with_space(sent):
    en_tags = ['<e1>', '</e1>', '<e2>', '</e2>']
    for tag in en_tags:
        s_i = sent.index(tag)
        if sent[s_i+len(tag)] != ' ':
            sent = sent[:s_i+len(tag)] + ' ' + sent[s_i+len(tag):]
        if sent[s_i-1] != ' ':
            sent = sent[:s_i] + ' ' + sent[s_i:]
    return sent


def turn_char_ids_to_word_ids(sent_words, en1p, en2p):
    word_lengths = list(map(len, sent_words))
    word_lengths_cumsum = [0]
    for w_i in range(0, len(word_lengths)):
        word_lengths_cumsum.append(word_lengths_cumsum[-1] + word_lengths[w_i])
    new_en1start = -1
    if en1p[0] in word_lengths_cumsum:
        new_en1start = word_lengths_cumsum.index(en1p[0])
    else:
        for j in range(1, len(word_lengths_cumsum)):
            if word_lengths_cumsum[j - 1] < en1p[0] < word_lengths_cumsum[j]:
                if en1p[0] - word_lengths_cumsum[j - 1] <= word_lengths_cumsum[j] - en1p[0]:
                    new_en1start = j - 1
                else:
                    new_en1start = j
    new_en2start = -1
    if en2p[0] in word_lengths_cumsum:
        new_en2start = word_lengths_cumsum.index(en2p[0])
    else:
        for j in range(1, len(word_lengths_cumsum)):
            if word_lengths_cumsum[j - 1] < en2p[0] < word_lengths_cumsum[j]:
                if en2p[0] - word_lengths_cumsum[j - 1] <= word_lengths_cumsum[j] - en2p[0]:
                    new_en2start = j - 1
                else:
                    new_en2start = j
    new_en1end = -1
    if en1p[1] + 1 in word_lengths_cumsum:
        new_en1end = word_lengths_cumsum.index(en1p[1] + 1) - 1
    else:
        for j in range(1, len(word_lengths_cumsum)):
            if word_lengths_cumsum[j - 1] < en1p[1] + 1 < word_lengths_cumsum[j]:
                if en1p[1] + 1 - word_lengths_cumsum[j - 1] <= word_lengths_cumsum[j] + 1 - en1p[1]:
                    new_en1end = j - 2
                else:
                    new_en1end = j - 1
    new_en2end = -1
    if en2p[1] + 1 in word_lengths_cumsum:
        new_en2end = word_lengths_cumsum.index(en2p[1] + 1) - 1
    else:
        for j in range(1, len(word_lengths_cumsum)):
            if word_lengths_cumsum[j - 1] < en2p[1] + 1 < word_lengths_cumsum[j]:
                if en2p[1] + 1 - word_lengths_cumsum[j - 1] <= word_lengths_cumsum[j] + 1 - en2p[1]:
                    new_en2end = j - 2
                else:
                    new_en2end = j - 1
    return (new_en1start, new_en1end), (new_en2start, new_en2end)


def process_original_sentence(sentence, spacy_model, if_keep_en_tag, max_length=None):
    """这里要注意的是max_length不包括entity tag，即如果保留entity tag，最终max_length会+=4"""
    sentence_without_ent_tag, ((e1chr_s, e1chr_e), (e2chr_s, e2chr_e)) = remove_ent_mask(sentence)
    pipeline = spacy_model(sentence_without_ent_tag)

    tokens = list(map(lambda x: x.text, pipeline))
    (e1s, e1e), (e2s, e2e) = turn_char_ids_to_word_ids(tokens, (e1chr_s, e1chr_e), (e2chr_s, e2chr_e))

    # ["I", "have", "a", "dog"] -> ["▁I", "▁have", "▁a", "▁dog"]
    tokens = turn_tokens_to_transformer_input_style(tokens, sentence_without_ent_tag)

    dep_heads = []
    # pos_tags = []
    for t, token in enumerate(pipeline):
        assert t == token.i
        # pos_tags.append(token.pos_)
        dep_heads.append(token.head.i)

    if not max_length:
        max_length = len(pipeline)

    if if_keep_en_tag:
        max_length += 4
        tokens.insert(e2e + 1, '</e2>')
        tokens.insert(e2s, '<e2>')
        tokens.insert(e1e + 1, '</e1>')
        tokens.insert(e1s, '<e1>')
        dep_heads.insert(e2e + 1, dep_heads[e2e])
        dep_heads.insert(e2s, dep_heads[e2s])
        dep_heads.insert(e1e + 1, dep_heads[e1e])
        dep_heads.insert(e1s, dep_heads[e1s])
        e1s += 1
        e1e += 1
        e2s += 3
        e2e += 3

    dep_fw = np.zeros([max_length, max_length]).astype(np.int)
    for t in range(len(dep_fw)):
        dep_fw[dep_heads[t]][t] = 1
        dep_fw[t][t] = 1  # 自环

    assert len(tokens) == len(dep_fw)
    # dep_bw = np.transpose().copy()

    return tokens, dep_fw, ((e1s, e1e), (e2s, e2e))


def turn_tokens_to_transformer_input_style(tokens, original_sentence):
    # ["I", "have", "a", "dog"] -> ["▁I", "▁have", "▁a", "▁dog"]
    assert sum(map(len, tokens)) == len(original_sentence.replace(" ", ""))
    n = len(original_sentence)
    p = 0
    transformer_style_tokens = list()
    transformer_style_tokens.append("▁" + tokens[0])
    p += len(tokens[0])
    for token_id in range(1, len(tokens)):
        while p < n and original_sentence[p] != tokens[token_id][0]:
            p += 1
        if p >= n:
            break
        if original_sentence[p - 1] == ' ':
            if 97 <= ord(tokens[token_id][0]) <= 122 or 65 <= ord(tokens[token_id][0]) <= 90 or 48 <= ord(
                    tokens[token_id][0]) <= 57:
                transformer_style_tokens.append("▁" + tokens[token_id])
            else:
                transformer_style_tokens.append(tokens[token_id])
        else:
            transformer_style_tokens.append(tokens[token_id])
        p += len(tokens[token_id])
    assert len(transformer_style_tokens) == len(tokens)

    return transformer_style_tokens


def static_relations(relations):
    relation_count_dict = defaultdict(int)
    for rel in relations:
        relation_count_dict[rel] += 1
    # relation_set = sorted(relation_count_dict, key=relation_count_dict.__getitem__, reverse=True)
    relation_count_dict = sorted(relation_count_dict.items(), key=lambda x: x[1], reverse=True)
    relation_count_dict = {x[0]: x[1] for x in relation_count_dict}
    relation2id = {rel: r_i for r_i, rel in enumerate(relation_count_dict.keys())}
    print('relation number %d' % len(relation2id))
    return relation2id, relation_count_dict


def create_examples_for_xlnet(sents, rels, tokenizer, relation2id, keep_en_tag, use_type_ids, max_length,
                              use_attention_mask=True):
    assert len(sents) == len(rels)
    spacy_model = spacy.load(SpaCyModelName)
    input_ids_list = []
    type_ids_list = [] if use_type_ids else None
    attention_mask_list = [] if use_attention_mask else None
    en_position_list = []
    offset_list = []
    dependency_adj_fw_list = []
    dependency_adj_bw_list = []
    relation_id_list = []
    abandon_count = 0
    for s_i, sent in tqdm(enumerate(sents), desc="Data Pre-process"):

        sentence_without_ent_tag, ((e1chr_s, e1chr_e), (e2chr_s, e2chr_e)) = remove_ent_mask(sent)
        pipeline = spacy_model(sentence_without_ent_tag)

        tokens = list(map(lambda x: x.text, pipeline))
        current_len = len(tokens)
        if keep_en_tag:
            current_len += 6
        else:
            current_len += 2
        if current_len > max_length:
            abandon_count += 1
            continue
        (e1s, e1e), (e2s, e2e) = turn_char_ids_to_word_ids(tokens, (e1chr_s, e1chr_e), (e2chr_s, e2chr_e))
        if e1e < e1s or e2e < e2s:
            abandon_count += 1
            continue
        # ["I", "have", "a", "dog"] -> ["▁I", "▁have", "▁a", "▁dog"]
        tokens = turn_tokens_to_transformer_input_style(tokens, sentence_without_ent_tag)

        dep_heads = []
        # pos_tags = []
        for t, token in enumerate(pipeline):
            assert t == token.i
            # pos_tags.append(token.pos_)
            dep_heads.append(token.head.i)

        if keep_en_tag:
            tokens.insert(e2e + 1, '</e2>')
            tokens.insert(e2s, '<e2>')
            tokens.insert(e1e + 1, '</e1>')
            tokens.insert(e1s, '<e1>')
            dep_heads.insert(e2e + 1, dep_heads[e2e])
            dep_heads.insert(e2s, dep_heads[e2s])
            dep_heads.insert(e1e + 1, dep_heads[e1e])
            dep_heads.insert(e1s, dep_heads[e1s])
            e1s += 1
            e1e += 1
            e2s += 3
            e2e += 3

        # pad to max_length, left pad
        offset = max_length - len(tokens) - 2
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids.append(tokenizer.sep_token_id)
        token_ids.append(tokenizer.cls_token_id)
        input_ids = [tokenizer.pad_token_id] * offset + token_ids
        e1s += offset
        e1e += offset
        e2s += offset
        e2e += offset

        dep_fw = np.zeros([max_length, max_length]).astype(np.int)
        assert offset + len(dep_heads) + 2 == len(dep_fw)
        for t in range(len(dep_heads)):
            dep_fw[offset+dep_heads[t]][offset+t] = 1
            dep_fw[offset+t][offset+t] = 1  # 自环
        dep_fw[-1][-1] = 1
        dep_fw[-2][-2] = 1
        assert len(input_ids) == len(dep_fw) == max_length
        dep_bw = np.transpose(dep_fw).copy()

        input_ids_list.append(input_ids)
        relation_id_list.append(relation2id[rels[s_i]])
        en_position_list.append(((e1s, e1e), (e2s, e2e)))
        offset_list.append(offset)
        dependency_adj_fw_list.append(dep_fw)
        dependency_adj_bw_list.append(dep_bw)
        if use_attention_mask:
            attention_mask = [0] * max_length
            attention_mask[offset:] = [1] * (max_length - offset)
            attention_mask_list.append(attention_mask)
        if use_type_ids:
            type_ids = [0] * max_length
            type_ids[-1] = 2
            type_ids[e1s: e1e+1] = [1] * (e1e+1-e1s)
            type_ids[e2s: e2e+1] = [1] * (e2e+1-e2s)
            type_ids_list.append(type_ids)
    assert len(input_ids_list) == len(relation_id_list) == len(en_position_list) == \
        len(dependency_adj_fw_list)
    if abandon_count > 0:
        print("Due to over length, %d examples have been abandoned." % abandon_count)
    return (input_ids_list, attention_mask_list, type_ids_list, relation_id_list, offset_list, en_position_list,
            dependency_adj_fw_list, dependency_adj_bw_list)


class PreTrainGraphDataset(Dataset):
    def __init__(self, examples, args):
        super(PreTrainGraphDataset, self).__init__()
        self.input_ids_list = examples[0]
        self.attention_mask_list = examples[1]
        self.type_ids_list = examples[2]
        self.relation_id_list = examples[3]
        self.offset_list = examples[4]
        self.en_position_list = examples[5]
        self.np_dep_adj_fw_list = examples[6]
        self.np_dep_adj_bw_list = examples[7]

        self.use_type_ids = True if args.entity_position_encoding == "token_type_ids" else False

    def __getitem__(self, index):
        if self.use_type_ids and self.type_ids_list is not None:
            return self.input_ids_list[index], self.attention_mask_list[index], self.type_ids_list[index], self.relation_id_list[index], self.offset_list[index], self.en_position_list[index], self.np_dep_adj_fw_list[index], self.np_dep_adj_bw_list[index]
        else:
            return self.input_ids_list[index], self.attention_mask_list[index], self.relation_id_list[index], self.offset_list[index], self.en_position_list[index], self.np_dep_adj_fw_list[index], self.np_dep_adj_bw_list[index]

    def __len__(self):
        return len(self.input_ids_list)


def collate_function(batch_data_list):
    b_input_ids = torch.LongTensor([item[0] for item in batch_data_list])
    b_attn_mask = torch.LongTensor([item[1] for item in batch_data_list])
    if len(batch_data_list) == 8:
        b_type_ids = torch.LongTensor([item[2] for item in batch_data_list])
    b_rel_ids = torch.LongTensor([item[-5] for item in batch_data_list])
    b_offsets = [item[-4] for item in batch_data_list]
    b_en_position = [item[-3] for item in batch_data_list]
    seq_length = b_input_ids.shape[1]
    b_adj_fw = torch.FloatTensor(np.concatenate([item[-2].reshape(1, seq_length, seq_length) for item in batch_data_list], axis=0))
    b_adj_bw = torch.FloatTensor(np.concatenate([item[-1].reshape(1, seq_length, seq_length) for item in batch_data_list], axis=0))
    if len(batch_data_list) == 8:
        return b_input_ids, b_attn_mask, b_type_ids, b_rel_ids, b_offsets, b_en_position, b_adj_fw, b_adj_bw
    else:
        return b_input_ids, b_attn_mask, None, b_rel_ids, b_offsets, b_en_position, b_adj_fw, b_adj_bw


class FakeArgs:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            self.__setattr__(k, v)


fake_args_dict = {
    "dataset": "kbp37",
    "data_cache_dir": "./data/cache/",
    "kbp37_split_dev": False,
    "entity_position_encoding": "token_type_ids",
    "model_architecture": "joint",
    "aggregation_way": "entity_start",
    "graph_pool_fun": "max",
    "gcn_layer": 2,
    "max_seq_length": 180,
    "batch_size": 4,
    "xlnet_learning_rate": 3e-5,
    "downstream_learning_rate": 3e-5,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "num_train_epochs": 2,
    "no_cuda": True,
    "seed": 123,
    "gradient_accumulation_steps": 1
}
