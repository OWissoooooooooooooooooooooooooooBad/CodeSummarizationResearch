# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import copy
import bleu
import torch
import random
import logging
import math
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from utils import get_model_size, Collator, Dataset, embedding_code, DoNothingCollator, EvalDataset
from model import FiDT5, Retriever
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          T5Config, T5ForConditionalGeneration, RobertaTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    
    t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)
    
    model.to(args.device)   
    
    collator = Collator(tokenizer, text_maxlength=args.max_source_length, answer_maxlength=args.max_target_length)
    
    # build retriever
    Rtokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
    Rconfig = RobertaConfig.from_pretrained('microsoft/unixcoder-base')
    Rmodel = RobertaModel.from_pretrained('microsoft/unixcoder-base')
    retriever = Retriever(Rmodel)
    retriever = retriever.to(args.device)
    
    
    if args.n_gpu > 1:
        # multi-gpu training
        model = nn.DataParallel(model)
        retriever = nn.DataParallel(retriever)
    Umodel = model.module if hasattr(model, 'module') else model
    Uretriever = retriever.module if hasattr(retriever, 'module') else retriever

    if args.do_train:
        n_passage = 4
        code_vec = None
        
        # Prepare training data loader
        train_data = Dataset(args.train_filename, Rtokenizer)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps, collate_fn=DoNothingCollator())
        

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        Roptimizer = AdamW(retriever.parameters(), lr=args.learning_rate, eps=1e-8)
        Rscheduler = get_linear_schedule_with_warmup(Roptimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)
    
        #Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        Umodel.overwrite_forward_crossattention()
        Umodel.reset_score_storage() 
        model.train()
        patience, best_bleu, losses, dev_dataset = 0, 0, [], {}
        update_tokens, update_idx = [], []
        for epoch in range(args.num_train_epochs):
            if code_vec is not None:
                del(code_vec)
            logger.info("***** Embedding code *****")
            code_vec = embedding_code(train_data.data, retriever, args)
            for idx,batch in enumerate(train_dataloader):
                
                #retrieve similar code
                query_ids = [x['source_ids'] for x in batch]
                query_ids = torch.tensor(query_ids).to(args.device)
                query_vec = retriever(code_inputs = query_ids)
                with torch.no_grad():
                    query_vec_np = torch.zeros_like(query_vec)
                    query_vec_np += query_vec
                    query_vec_np = query_vec_np.cpu().numpy()
                scores = np.matmul(query_vec_np, code_vec.T)
                sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
                for i in range(len(batch)):
                    similar_code = []
                    similar_comment = []
                    similar = []
                    for j in range(n_passage):
                        similar_code.append(train_data.data[sort_ids[i][j+1]]['source'])
                        similar_comment.append(train_data.data[sort_ids[i][j+1]]['target'])
                        similar.append(int(train_data.data[sort_ids[i][j+1]]['idx']))
                    batch[i]['similar_code']=similar_code
                    batch[i]['similar_comment']=similar_comment
                    batch[i]['similar']=similar
                del(scores)
                del(sort_ids)
                
                (idx, labels, _, context_ids, context_mask) = collator(batch)
                
                
                # train reader
                Umodel.reset_score_storage()
                loss = model(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    labels=labels.cuda()
                )[0]
                crossattention_scores = Umodel.get_crossattention_scores(context_mask.cuda()).detach()

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
                
                # train retriever
                similar_ids = [y for x in batch for y in x['similar'] ]
                update_idx.append(similar_ids)
                key_ids = [[train_data.data[y]['source_ids'] for y in x['similar']] for x in batch]
                key_ids = torch.tensor(key_ids).to(args.device)
                key_ids = key_ids.view(len(batch) * n_passage, -1)
                update_tokens.append(key_ids)
                key_vec = retriever(code_inputs = key_ids)
                key_vec = key_vec.view(len(batch), n_passage, -1)
                score = torch.einsum(
                            'bd,bid->bi',
                            query_vec,
                            key_vec
                        )
                score = score / np.sqrt(query_vec.size(-1))
            
                Rloss = Uretriever.kldivloss(score, crossattention_scores)
                if args.n_gpu > 1:
                    Rloss = Rloss.mean()
                    
                if args.gradient_accumulation_steps > 1:
                    Rloss = Rloss / args.gradient_accumulation_steps
                
                Rloss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    Roptimizer.step()
                    Roptimizer.zero_grad()
                    Rscheduler.step()
                    with torch.no_grad():
                        for key_ids, similar_ids in zip(update_tokens, update_idx):
                            key_vec = retriever(code_inputs=key_ids)
                            key_vec = key_vec.cpu().numpy()
                            for i, idx in enumerate(similar_ids):
                                code_vec[idx] = key_vec[i]
                        update_tokens, update_idx = [], []
                
            if args.do_eval:
                #Calculate bleu  
                if code_vec is not None:
                    del(code_vec)
                logger.info("***** Embedding code *****")
                code_vec = embedding_code(train_data.data, retriever, args)
                if 'dev_bleu' in dev_dataset:
                    eval_data=dev_dataset['dev_bleu']
                else:
                    eval_data = EvalDataset(args.dev_filename, Rtokenizer, retriever, code_vec, args, n_passage, train_data.data)
                    dev_dataset['dev_bleu'] = eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collator)

                model.eval() 
                pred_ids = []
                for batch in eval_dataloader:
                    (idx, _, _, context_ids, context_mask) = batch
                    with torch.no_grad():
                        if args.n_gpu > 1:
                            preds = model.module.generate(
                                input_ids=context_ids.cuda(),
                                attention_mask=context_mask.cuda(),
                                max_length=args.max_target_length
                            )
                        else:
                            preds = model.generate(
                                input_ids=context_ids.cuda(),
                                attention_mask=context_mask.cuda(),
                                max_length=args.max_target_length
                            )
                        top_preds = list(preds.cpu().numpy())
                    pred_ids.extend(top_preds)
                
                pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
                
                model.train()
                predictions = []
                with open(args.output_dir+"/dev.output",'w') as f, open(args.output_dir+"/dev.gold",'w') as f1:
                    for ref,gold in zip(pred_nls,eval_data.data):
                        predictions.append(str(gold["idx"])+'\t'+ref)
                        f.write(str(gold["idx"])+'\t'+ref+'\n')
                        f1.write(str(gold["idx"])+'\t'+gold["target"]+'\n')     

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    retriever_to_save = retriever.modules if hasattr(retriever, 'module') else retriever
                    output_retriever_file = os.path.join(output_dir, "pytorch_retriever.bin")
                    torch.save(retriever_to_save.state_dict(), output_retriever_file)
                    patience =0
                else:
                    patience +=1
                    if patience ==2:
                        break
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))                

        eval_data = Dataset(args.test_filename)  

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collator)

        model.eval() 
        pred_ids = []
        for batch in tqdm(eval_dataloader):
            (idx, _, _, context_ids, context_mask) = batch
            with torch.no_grad():
                if args.n_gpu > 1:
                    preds = model.module.generate(
                        input_ids=context_ids.cuda(),
                        attention_mask=context_mask.cuda(),
                        max_length=args.max_target_length
                    )
                else:
                    preds = model.generate(
                        input_ids=context_ids.cuda(),
                        attention_mask=context_mask.cuda(),
                        max_length=args.max_target_length
                    )
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
        
        pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
                    
        model.train()
        predictions=[]
        with open(args.output_dir+"/test.output",'w') as f, open(args.output_dir+"/test.gold",'w') as f1:
            for ref,gold in zip(pred_nls,eval_data.data):
                predictions.append(str(gold["idx"])+'\t'+ref)
                f.write(str(gold["idx"])+'\t'+ref+'\n')
                f1.write(str(gold["idx"])+'\t'+gold["target"]+'\n')     

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test.gold")) 
        dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    
                
if __name__ == "__main__":
    main()

