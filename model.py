#!/usr/bin/env/ python
# coding: utf-8
"""
@File: model.py
@Description: Void
@Copyright: Zhuoxian Wang
@Email: sysu14336205@163.com
@Date: 2020/3/26
"""
from transformers import XLNetModel
import torch
from torch import nn
import math

from const import PreTrainedXLNetModelPath


class GCN(nn.Module):
    def __init__(self, hid_size=1024, concatenate=False):
        super(GCN, self).__init__()

        self.hid_size = hid_size
        # self.W = nn.Parameter(torch.FloatTensor(self.hid_size, self.hid_size // 2).cuda())
        # self.b = nn.Parameter(torch.FloatTensor(self.hid_size // 2, ).cuda())
        self.concatenate = concatenate
        if concatenate:
            self.linear = nn.Linear(self.hid_size, self.hid_size // 2, bias=True)
        else:
            self.linear = nn.Linear(self.hid_size, self.hid_size, bias=True)
        # self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.hid_size // 2)

        # self.W.data.uniform_(-stdv, stdv)
        # self.b.data.uniform_(-stdv, stdv)
        self.linear.weight.data.uniform_(-stdv, stdv)
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj, is_relu=True, average=False):
        # out = torch.matmul(inp, self.W) + self.b
        out = self.linear(inp)
        out = torch.matmul(adj, out)        # [bs, max_seq_len, 1024]

        if average:
            margin_sum = torch.sum(adj, dim=2)
            margin_sum[margin_sum == 0] = 1
            out = out / margin_sum.unsqueeze(2)

        if is_relu:
            # out = nn.functional.relu(out)
            out = nn.functional.leaky_relu(out)

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(hid_size=%d)' % self.hid_size


class GraphPreTrain(nn.Module):
    def __init__(self, args, drop_out=0.5):
        super(GraphPreTrain, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(PreTrainedXLNetModelPath)
        self.hidden_size = self.xlnet.config.hidden_size

        self.rel_feature_size = args.rel_feature_size
        self.num_rel = args.num_rel
        self.gcn_average = True if args.gcn_average else False
        self.gcn_concatenate = True if args.gcn_concatenate else False

        # define model architecture and the way of sequence aggregation
        if args.model_architecture:
            assert args.model_architecture in ["joint", "graph_alone", "sequence_alone"]
            self.model_architecture = args.model_architecture
        else:
            self.model_architecture = "joint"
        if self.model_architecture != "graph_alone":
            self.linear_s = nn.Linear(self.hidden_size * 2, self.rel_feature_size)
            self.batch_norm_s = nn.BatchNorm1d(num_features=2*self.hidden_size)
        if self.model_architecture != "sequence_alone":
            self.linear_g = nn.Linear(self.hidden_size * 2, self.rel_feature_size)
            self.batch_norm_g = nn.BatchNorm1d(num_features=2 * self.hidden_size)
        self.linear_rel = nn.Linear(self.rel_feature_size, self.num_rel)

        self.entity_position_encoding = args.entity_position_encoding
        if args.aggregation_way:
            assert args.aggregation_way in ["cls", "entity_start", "max_pool"]
            self.aggregation_way = args.aggregation_way
        else:
            self.aggregation_way = "entity_start"

        self.graph_pool_function = "max" if not args.graph_pool_fun else args.graph_pool_fun  # max or mean

        self.gcn_layer = args.gcn_layer
        self.hidden_size = self.xlnet.config.hidden_size

        self.gcn_fw = nn.ModuleList([GCN(self.hidden_size, self.gcn_concatenate) for _ in range(self.gcn_layer)])
        self.gcn_bw = nn.ModuleList([GCN(self.hidden_size, self.gcn_concatenate) for _ in range(self.gcn_layer)])

        self.drop_out = nn.Dropout(drop_out)

    def forward(self, input_ids, attn_mask, token_type_ids, dep_fw, dep_bw, en_positions):
        """
        token_type_ids == None if entity encoding way is not 'token_type_ids'
        for input ['<pad>', '<pad>', 'a', 'e1', 'b', 'e2_0', 'e2_1',...]
        en_position = (3, 3), (5, 6)
        """
        batch_size = input_ids.shape[0]

        transformers_out = self.xlnet(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids     # may be None
        )[0]

        if self.aggregation_way == 'cls':
            sequence_features = torch.cat((transformers_out[:, -1, :], transformers_out[:, -1, :]), dim=1)
        elif self.aggregation_way == "max_pool":
            en1_seq_feature_list = list()
            en2_seq_feature_list = list()
            for batch_id in range(batch_size):
                en1_position = en_positions[batch_id][0]
                en2_position = en_positions[batch_id][1]
                en1_seq_feature = transformers_out[batch_id, en1_position[0]:en1_position[1] + 1].max(dim=0, keepdim=True)[0]
                en1_seq_feature_list.append(en1_seq_feature)
                en2_seq_feature = transformers_out[batch_id, en2_position[0]:en2_position[1] + 1].max(dim=0, keepdim=True)[0]
                en2_seq_feature_list.append(en2_seq_feature)
            en1_seq_features = torch.cat(en1_seq_feature_list, dim=0)
            en2_seq_features = torch.cat(en2_seq_feature_list, dim=0)
            sequence_features = torch.cat((en1_seq_features, en2_seq_features), dim=1)   # [bs, 2 * 1024]
        elif self.aggregation_way == "entity_start":
            en1_seq_feature_list = list()
            en2_seq_feature_list = list()
            for batch_id in range(batch_size):
                en1_position = en_positions[batch_id][0]
                en2_position = en_positions[batch_id][1]
                entity_start_pos1 = en1_position[0] - 1 if self.entity_position_encoding == "entity_tag" else \
                    en1_position[0]
                en1_seq_feature = transformers_out[batch_id, entity_start_pos1].unsqueeze(0)
                en1_seq_feature_list.append(en1_seq_feature)
                entity_start_pos2 = en2_position[0] - 1 if self.entity_position_encoding == "entity_tag" else \
                    en2_position[0]
                en2_seq_feature = transformers_out[batch_id, entity_start_pos2].unsqueeze(0)
                en2_seq_feature_list.append(en2_seq_feature)
            en1_seq_features = torch.cat(en1_seq_feature_list, dim=0)
            en2_seq_features = torch.cat(en2_seq_feature_list, dim=0)
            sequence_features = torch.cat((en1_seq_features, en2_seq_features), dim=1)   # [bs, 2 * 1024]
        else:
            raise ValueError("Aggregation way error!")

        if self.model_architecture != "sequence_alone":
            out = transformers_out
            for i in range(self.gcn_layer):
                out_fw = self.gcn_fw[i](out, dep_fw, average=self.gcn_average)
                out_bw = self.gcn_bw[i](out, dep_bw, average=self.gcn_average)
                if self.gcn_concatenate:
                    out = torch.cat([out_fw, out_bw], dim=2)
                else:
                    out = out_fw + out_bw
                out = self.drop_out(out)          # [bs, max_len, 1024]
            en1_graph_feature_list = list()
            en2_graph_feature_list = list()
            for batch_id in range(batch_size):
                en1_position = en_positions[batch_id][0]
                en2_position = en_positions[batch_id][1]
                if self.graph_pool_function == "max":
                    en1_graph_feature = out[batch_id, en1_position[0]:en1_position[1]+1].max(dim=0, keepdim=True)[0]
                    en1_graph_feature_list.append(en1_graph_feature)
                    en2_graph_feature = out[batch_id, en2_position[0]:en2_position[1]+1].max(dim=0, keepdim=True)[0]
                    en2_graph_feature_list.append(en2_graph_feature)
                else:
                    en1_graph_feature = out[batch_id, en1_position[0]:en1_position[1]+1].mean(dim=0, keepdim=True)
                    en1_graph_feature_list.append(en1_graph_feature)
                    en2_graph_feature = out[batch_id, en2_position[0]:en2_position[1]+1].mean(dim=0, keepdim=True)
                    en2_graph_feature_list.append(en2_graph_feature)
            en1_graph_features = torch.cat(en1_graph_feature_list, dim=0)
            en2_graph_features = torch.cat(en2_graph_feature_list, dim=0)

            graph_features = torch.cat((en1_graph_features, en2_graph_features), dim=1)  # [bs, 2 * 1024]

        if self.model_architecture == "sequence_alone":
            # sequence alone
            relation_features = self.linear_s(self.batch_norm_s(sequence_features))
        elif self.model_architecture == "graph_alone":
            relation_features = self.linear_g(self.batch_norm_g(graph_features))
        else:
            relation_features = self.linear_s(self.batch_norm_s(sequence_features)) + self.linear_g(self.batch_norm_g(graph_features))
        scores = self.linear_rel(nn.functional.leaky_relu(relation_features))

        return scores
