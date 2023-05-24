# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import numpy as np
import transformers
from torch import nn
import torch
import torch.nn.functional as F
import json
 
def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))       

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            if hasattr(mod, 'module'):
                block.append(mod.module)
            else:
                block.append(mod)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages*passage_length, -1)
        return outputs
    
def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, tokenizer, text_maxlength=500, answer_maxlength=64):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        if batch[0]['target'] != None:
            index = torch.tensor([ex['idx'] for ex in batch])
            target = [ex['target'] for ex in batch]
            target = self.tokenizer.batch_encode_plus(
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False,
            )
            target_ids = target["input_ids"]
            target_mask = target["attention_mask"].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)
        else:
            target_ids = None
            target_mask = None

        def append_question(example):
            if example['similar_code'] is None:
                return [example['source']]
            return [example['source'] + "</s>" + x + "</s>" + y for x,y in zip(example['similar_comment'], example['similar_code'])]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            similar_code = []
            similar_comment = []
            for i in range(int(js['num'])):
                similar_code.append(js['similar_code_{}'.format(i)])
                similar_comment.append(js['similar_comment_{}'.format(i)])
            examples.append(
                {
                    "idx" : idx,
                    "source" : js['source_code'],
                    "target" : js['source_comment'],
                    "similar_code" : similar_code,
                    "similar_comment" : similar_comment
                } 
            )
    return examples

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.data = read_examples(filename)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]