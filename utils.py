# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import numpy as np
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset
 
def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))       

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
    
class DoNothingCollator(object):
    def __init__(self):
        super(DoNothingCollator)
    def __call__(self, batch):
        return batch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer):
        self.data = read_examples(filename, tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

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

def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0
    
def read_examples(filename, tokenizer):
    examples = []
    code_length = 256
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())
            code_tokens = tokenizer.tokenize(code)[:code_length-4]
            code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            padding_length = code_length - len(code_ids)
            code_ids += [tokenizer.pad_token_id]*padding_length
            examples.append({
                'idx':js['idx'],
                'source':code,
                'source_ids':code_ids,
                'target':nl
            })
    return examples

class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i]["source_ids"])

def embedding_code(examples, retriever, args):
    code_vecs = []
    dataset = TextDataset(examples)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            code_vec=retriever(code_inputs=batch.to(args.device))
            code_vecs.append(code_vec.cpu().numpy())
    code_vecs = np.concatenate(code_vecs,0)
    return code_vecs

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer, retriever, train_vec, args, n_passage, train_data):
        self.data = read_examples(filename, tokenizer)
        eval_vecs = embedding_code(self.data, retriever, args)
        scores = np.matmul(eval_vecs, train_vec.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
        for i in range(len(sort_ids)):
            similar_code = []
            similar_comment = []
            for k in range(n_passage):
                similar_code.append(train_data[sort_ids[i][k]]['source'])
                similar_comment.append(train_data[sort_ids[i][k]]['target'])
            self.data[i]['similar_code'] = similar_code
            self.data[i]['similar_comment'] = similar_comment
        del(eval_vecs)
        del(scores)
        del(sort_ids)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]