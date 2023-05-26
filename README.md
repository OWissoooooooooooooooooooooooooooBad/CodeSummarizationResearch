# CodeSummarizationResearch
围绕代码注释自动生成做的一些探索

| 任务 | 进度 | BLEU(20.39) |
| --- | --- | --- |
| 将CodeT5中关于CodeSummarization的部分提取出来，并保证运行结果与直接使用原版本一致 | ✔ | 20.5 |
| 增加检索模块 | ✔ | 22.45 |
| 增加FiD模块 | ✔ | 22.04 |
| 测试联合训练 | ❌ |  |
| 多GPU训练 | ❌ | - |

## 数据
来源于Microsoft CodeBert项目
```
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/ruby.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```
在我的项目中，只使用了其中java的代码，并将其放在`dataset/train.jsonl`, `dataset/test.jsonl`, `dataset/valid.jsonl`三个文件中。如果想测试其他语言，下载后更改各个filename即可。

## 训练
```
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path Salesforce/codet5-base \
	--train_filename data/train_DR.jsonl \
	--dev_filename data/valid_DR.jsonl \
	--output_dir saved_models/retrieval \
	--max_source_length 512 \
	--max_target_length 64 \
	--beam_size 10 \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
```
## 测试
```
python run.py \
	--do_test \
	--model_name_or_path Salesforce/codet5-base \
	--test_filename data/test_DR.jsonl \
	--output_dir saved_models/retrieval \
	--max_source_length 512 \
	--max_target_length 64 \
	--beam_size 10 \
	--train_batch_size 24 \
	--eval_batch_size 24 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
```