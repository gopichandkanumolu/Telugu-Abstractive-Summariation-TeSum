```python
!pip install git+https://github.com/huggingface/transformers --q
```


```python
!pip install -r "/kaggle/input/huggingfacetransformers/transformers/examples/pytorch/summarization/requirements.txt" --q
```


```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

## Data


```python
import pandas as pd
train_df=pd.read_csv('/kaggle/input/tesum-dataset/TeSum_train_data.csv')
train_df.head(1)
```

## Fine-tuning


```python
!python '/kaggle/input/huggingfacetransformers/transformers/examples/pytorch/summarization/run_summarization_indicbart.py' \
    --model_name_or_path ai4bharat/IndicBART \
    --do_train True\
    --do_eval True\
    --do_predict True\
    --train_file '/kaggle/input/tesum-dataset/TeSum_train_data.csv' \
    --validation_file '/kaggle/input/tesum-dataset/TeSum_dev_data.csv' \
    --test_file  '/kaggle/input/tesum-dataset/TeSum_test_data.csv' \
    --num_train_epochs 10 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --source_prefix "summarize: " \
    --text_column "cleaned_text" \
    --summary_column "summary" \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --save_safetensors False \
    --output_dir "./model_checkpoints" \
    --overwrite_output_dir True\
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=6 \
    --max_source_length 512 \
    --max_target_length 256 \
    --val_max_target_length 256 \
    --num_beams 4 \
    --predict_with_generate | tee logs.txt
```


```python
#     --max_train_samples 10 \
#     --max_eval_samples 10 \
#     --max_predict_samples 10 \
```

## Testing


```python
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer

# Load the trained model and tokenizer
model_path = "/kaggle/input/tesum-indicbart-checkpoints/model_checkpoints/checkpoint-27160"
tokenizer = AlbertTokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)

# Define a function for summarization
def generate_summary(article_text, max_length=150):
    inputs = tokenizer.encode(article_text+' </s> <2te>', return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.



```python
# # Example usage
# article = ""
# summary = generate_summary(article)
# print("Generated Summary:", summary)
```


```python

```
