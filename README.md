# CodeSummarizationResearch
围绕代码注释自动生成做的一些探索

| 任务 | 进度 | BLEU(20.39) |
| --- | --- | --- |
| 将CodeT5中关于CodeSummarization的部分提取出来，并保证运行结果与直接使用原版本一致 | ✔ | 20.5 |
| 增加检索模块 | ❌ | |
| 增加FiD模块 | ❌ | |

## 训练
```
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path Salesforce/codet5-base \
	--train_filename data/train.jsonl \
	--dev_filename data/valid.jsonl \
	--output_dir saved_models \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 24 \
	--eval_batch_size 24 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
```
## 测试
```
python run.py \
	--do_test \
	--model_name_or_path Salesforce/codet5-base \
	--test_filename data/test.jsonl \
	--output_dir saved_models \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 24 \
	--eval_batch_size 24 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
```