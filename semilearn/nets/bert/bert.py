# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import BertModel
import os

class ClassificationBert(nn.Module):
    def __init__(self, name, num_classes=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained(name)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, num_classes)
        ])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        
        if only_feat:
            return pooled_output
        
        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output, 'hidden_states': out_dict['hidden_states']}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
            
        return result_dict
        
        
    def extract(self, x):
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}bert.embeddings'.format(prefix), blocks=r'^{}bert.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []

# Define multitask model
class MTLClassificationBert(nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(name)
        self.dropout = nn.Dropout(0.1)
        self.num_classes = num_classes

        if type(num_classes) != list:
            raise ValueError('num_classes should be a list in MTL setting')

        for num in self.num_classes:
            idx = num_classes.index(num)
            #setattr(self, f'classifier{idx}', nn.Linear(self.bert.config.hidden_size, num))
            setattr(self, f'classifier{idx}', nn.Sequential(*[
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, num)
        ]))

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):

        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)

        # [CLS] the beginning of the sequence can be used directly as a pooled representation as in BERT paper suggested
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)

        logits_ = []
        pooled_output_ = []
        for num in self.num_classes:
            idx = self.num_classes.index(num)
            classifier = getattr(self, f'classifier{idx}')
            logits = classifier(pooled_output)
            logits_.append(logits)
            pooled_output_.append(pooled_output)

        # FIXME: pooled_output_ and feat
        result_dict = {'logits': logits_, 'feat': pooled_output_, 'hidden_states': out_dict['hidden_states']}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]

        return result_dict

def bert_base_cased(pretrained=True, pretrained_path=None, **kwargs):
    model = ClassificationBert(name='bert-base-cased', **kwargs)
    return model


def bert_base_uncased(pretrained=True, pretrained_path=None, **kwargs):
    model = ClassificationBert(name='bert-base-uncased', **kwargs)
    return model

# FIXME
def biobert(num_classes, pretrained=True, pretrained_path=None, use_mtl=False):
    if use_mtl:
        model = MTLClassificationBert(name='dmis-lab/biobert-v1.1', num_classes=num_classes)
    else:
        model = ClassificationBert(name='dmis-lab/biobert-v1.1', num_classes=num_classes)
    return model

def pubmedbert(num_classes, pretrained=True, pretrained_path=None, use_mtl=False):
    if use_mtl:
        model = MTLClassificationBert(name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', num_classes=num_classes)
    else:
        model = ClassificationBert(name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', num_classes=num_classes)
    return model


