#!/usr/bin/env/ python
# coding: utf-8
"""
@File: main.py
@Description: Void
@Copyright: Zhuoxian Wang
@Email: sysu14336205@163.com
@Date: 2020/3/26
"""
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import logging
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm, trange
from transformers import XLNetTokenizer, AdamW
from sklearn.metrics import precision_score, f1_score, recall_score

from const import *
from utils import read_txt, create_examples_for_xlnet, static_relations, PreTrainGraphDataset, collate_function
from model import GraphPreTrain


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def create_logger(log_path: str=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path:
        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="semeval2010task8",
        type=str,
        help="semeval2010task8 or kbp37",
    )
    parser.add_argument(
        "--data_cache_dir",
        default="./data/cache/",
        type=str,
        help="directory to save data features",
    )
    parser.add_argument("--kbp37_split_dev", default=False, type=bool, help="If False, join dev set to train set.")
    parser.add_argument(
        "--entity_position_encoding",
        default="entity_tag",
        type=str,
        help="The way of entity position encoding, one of normal, entity_tag, and token_type_ids",
    )
    parser.add_argument(
        "--model_architecture",
        default="joint",
        type=str,
        required=True,
        help="joint, graph_alone or sequence_alone"
    )
    parser.add_argument(
        "--aggregation_way",
        default="entity_start",
        type=str,
        required=True,
        help="Aggregation way when extracting sequence feature, cls, entity_start or max_pool",
    )
    parser.add_argument(
        "--graph_pool_fun",
        default="max",
        type=str,
        help="max or mean",
    )
    parser.add_argument("--gcn_average", action="store_true", help="Do mean pool in GCN forward ")
    parser.add_argument("--gcn_concatenate", action="store_true", help="replace summation with concatenation in Bi-GCN")
    parser.add_argument(
        "--gcn_layer",
        default=2,
        type=int,
        help="Number of GCN layers."
    )
    parser.add_argument("--rel_feature_size", default=1024, type=int, help="Size of relation feature in final predict.")
    parser.add_argument(
        "--max_seq_length",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded. Will be set auto if not defined",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size when training.")
    parser.add_argument("--xlnet_learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--downstream_learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.",
    )    # epochs up to 10
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=123, type=int, help="Random seed.")

    args = parser.parse_args()
    if args.dataset == "kbp37":
        args.num_rel = 19
    elif args.dataset == "semeval2010task8":
        args.num_rel = 10
    else:
        raise ValueError("Invalid DataSet name!")

    # adjust max_seq_length
    if args.dataset == 'kbp37':
        if args.max_seq_length > 180:
            args.max_seq_length = 180
    else:
        if args.max_seq_length > 110:
            args.max_seq_length = 110
    return args


def load_and_cache_examples(args, tokenizer, logger, mode="train"):
    """SemEval2010Task8 does'not have dev set"""
    assert mode in ["train", "test", "dev"]

    if not os.path.exists(args.data_cache_dir):
        os.mkdir(args.data_cache_dir)

    cached_examples_file = os.path.join(
        args.data_cache_dir,
        "cached_{}_{}_{}_{}".format(args.dataset, mode, args.entity_position_encoding, str(args.max_seq_length)),
    )
    if os.path.exists(cached_examples_file):
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating features for %s %s set" % (args.dataset, mode))
        if args.dataset == 'kbp37':
            _, train_sentences, train_relations = read_txt(os.path.join(KBP37RawPath, "train.txt"))
            _, dev_sentences, dev_relations = read_txt(os.path.join(KBP37RawPath, "dev.txt"))
            _, test_sentences, test_relations = read_txt(os.path.join(KBP37RawPath, "test.txt"))
            if not args.kbp37_split_dev:
                train_sentences.extend(dev_sentences)
                train_relations.extend(dev_relations)
        else:
            _, train_sentences, train_relations = read_txt(os.path.join(SemEval2010Task8RawPath, "train.txt"))
            _, test_sentences, test_relations = read_txt(os.path.join(SemEval2010Task8RawPath, "test.txt"))
        relation2id_path = KBP37Relation2IdPath if args.dataset == "kbp37" else SemEval2010Relation2IdPath
        if os.path.exists(relation2id_path):
            with open(relation2id_path, 'r', encoding='utf8') as f:
                relation2id = json.load(f)
        else:
            relation2id, _ = static_relations(train_relations)
            with open(relation2id_path, 'w', encoding='utf8') as f:
                json.dump(relation2id, f)
        if mode == 'train':
            sentences, relations = train_sentences, train_relations
        elif mode == 'test':
            sentences, relations = test_sentences, test_relations
        else:
            if args.dataset == 'kbp37':
                sentences, relations = dev_sentences, dev_relations
            else:
                raise ValueError("SemEval2010Task8 does'not have dev set!")
        examples = create_examples_for_xlnet(sentences, relations, tokenizer, relation2id,
                                             True if args.entity_position_encoding=="entity_tag" else False,
                                             True if args.entity_position_encoding == "token_type_ids" else False,
                                             args.max_seq_length)
        torch.save(examples, cached_examples_file)
    return examples


def evaluate(model, data_iterator, device):
    prediction = []
    ground_true = []
    model.eval()
    for batch in data_iterator:
        b_input_ids, b_attn_mask, b_type_ids, b_rel_ids, b_offsets, b_en_position, b_adj_fw, b_adj_bw = batch
        b_input_ids = b_input_ids.to(device)
        b_attn_mask = b_attn_mask.to(device)
        b_type_ids = b_type_ids.to(device) if b_type_ids is not None else None
        b_adj_fw = b_adj_fw.to(device)
        b_adj_bw = b_adj_bw.to(device)
        scores = model(
            input_ids=b_input_ids,
            attn_mask=b_attn_mask,
            token_type_ids=b_type_ids,
            dep_fw=b_adj_fw,
            dep_bw=b_adj_bw,
            en_positions=b_en_position
        )
        prediction.extend(scores.to("cpu").argmax(dim=1).tolist())
        ground_true.extend(b_rel_ids.tolist())
    assert len(prediction) == len(ground_true)
    # print(list(zip(prediction, ground_true))[:100])
    return get_result(pred=prediction, label=ground_true)


def get_result(pred, label):
    micro_p = precision_score(label, pred, average='micro')
    macro_p = precision_score(label, pred, average='macro')
    weighted_p = precision_score(label, pred, average='weighted')

    micro_r = recall_score(label, pred, average='micro')
    macro_r = recall_score(label, pred, average='macro')
    weighted_r = recall_score(label, pred, average='weighted')

    micro_f1 = f1_score(label, pred, average='micro')
    macro_f1 = f1_score(label, pred, average='macro')
    weighted_f1 = f1_score(label, pred, average='weighted')

    result = {"micro_p": micro_p, "macro_p": macro_p, "weighted_p": weighted_p,
              "micro_r": micro_r, "macro_r": macro_r, "weighted_r": weighted_r,
              "micro_f1": micro_f1, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    return result


def main():
    args = get_args()

    # 方便调试
    """
    from utils import FakeArgs, fake_args_dict
    args = FakeArgs(fake_args_dict)
    if args.dataset == "kbp37":
        args.num_rel = NumRelKBP37
    elif args.dataset == "semeval2010task8":
        args.num_rel = NumRelSemEval2010Task8
    else:
        raise ValueError("Invalid DataSet name!")
    """

    set_seed(args)
    experiment_name = '_'.join([args.dataset, args.model_architecture, args.aggregation_way, args.entity_position_encoding])
    logger = create_logger(os.path.join(LoggingFilePath, experiment_name))

    tokenizer = XLNetTokenizer.from_pretrained(PreTrainedXLNetTokenizerPath)
    tokenizer.add_tokens(EntityTagList)

    # 加载数据
    train_examples = load_and_cache_examples(args, tokenizer, logger)
    test_examples = load_and_cache_examples(args, tokenizer, logger, mode="test")

    train_dataset = PreTrainGraphDataset(train_examples, args)
    test_dataset = PreTrainGraphDataset(test_examples, args)

    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function)
    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size//2, shuffle=False, collate_fn=collate_function)

    if args.dataset == "kbp37":
        args.num_rel = len(json.load(open(KBP37Relation2IdPath, 'r')))
        # args.num_rel = 19
    elif args.dataset == "semeval2010task8":
        args.num_rel = len(json.load(open(SemEval2010Relation2IdPath, 'r')))
        # args.num_rel = 10
    else:
        raise ValueError("Invalid DataSet name!")

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # load model
    model = GraphPreTrain(args)
    model.xlnet.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    # set optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "xlnet" in n],
            "lr": args.xlnet_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if "xlnet" not in n],
            "lr": args.downstream_learning_rate
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    loss_fun = torch.nn.CrossEntropyLoss().to(device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))   
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Learning Rate = {}, {}".format(args.xlnet_learning_rate, args.downstream_learning_rate))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()

    best_result_record_dict = defaultdict(int)
    epoch_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for epoch in epoch_iterator:
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_iterator), desc="Data iter"):
            model.train()
            b_input_ids, b_attn_mask, b_type_ids, b_rel_ids, b_offsets, b_en_position, b_adj_fw, b_adj_bw = batch
            b_input_ids = b_input_ids.to(device)
            b_attn_mask = b_attn_mask.to(device)
            b_type_ids = b_type_ids.to(device) if b_type_ids is not None else None
            b_adj_fw = b_adj_fw.to(device)
            b_adj_bw = b_adj_bw.to(device)
            b_rel_ids = b_rel_ids.to(device)
            scores = model(
                input_ids=b_input_ids,
                attn_mask=b_attn_mask,
                token_type_ids=b_type_ids,
                dep_fw=b_adj_fw,
                dep_bw=b_adj_bw,
                en_positions=b_en_position
            )
            loss = loss_fun(scores, b_rel_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
        
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
        print("Finish epoch %d training!"%epoch)
        model.zero_grad()
        epoch_result = evaluate(model, test_iterator, device)
        logger.info("Epoch %d: train loss=%.3f, micro f1=%.3f, macro f1=%.3f, weighted fa=%.3f" % (
            epoch,
            epoch_loss/len(train_iterator)*args.gradient_accumulation_steps,
            epoch_result['micro_f1'],
            epoch_result['macro_f1'],
            epoch_result['weighted_f1']
        ))
        worth_to_save = False
        for k, v in epoch_result.items():
            if v > best_result_record_dict[k]:
                best_result_record_dict[k] = v
                best_result_record_dict[k+'_epoch'] = epoch
                if 'f1' in k:
                    worth_to_save = True
        if worth_to_save:
            torch.save(model.state_dict(), os.path.join(ModelSavingPath, experiment_name+"_best.bin"))
    with open(os.path.join(ResultDumpPath, experiment_name + '.json'), 'w', encoding='utf8') as f:
        json.dump(best_result_record_dict, f)
    torch.save(model.state_dict(), os.path.join(ModelSavingPath, experiment_name + "_epoch%d.bin" % epoch))


if __name__ == '__main__':
    main()
