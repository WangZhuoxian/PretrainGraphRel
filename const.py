#!/usr/bin/env/ python
# coding: utf-8
"""
@File: const.py
@Description: Void
@Copyright: Zhuoxian Wang
@Email: sysu14336205@163.com
@Date: 2020/3/26
"""

PreTrainedXLNetModelPath = "./data/pre_train_xlnet/model"
PreTrainedXLNetTokenizerPath = "./data/pre_train_xlnet/tokenizer"

SpaCyModelName = "en_core_web_lg"

EntityTagList = ['<e1>', '</e1>', '<e2>', '</e2>']

KBP37RawPath = "./data/kbp37/raw"
SemEval2010Task8RawPath = "./data/SemEval2010_task8/raw"

NumRelKBP37 = 37
NumRelSemEval2010Task8 = 19
KBP37Relation2IdPath = "./data/kbp37/relation2id.json"
SemEval2010Relation2IdPath = "./data/SemEval2010_task8/relation2id.json"

LoggingFilePath = "./logging"
ResultDumpPath = "./results"
ModelSavingPath = "./saved_models"

