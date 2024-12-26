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
!python '/kaggle/input/huggingfacetransformers/transformers/examples/pytorch/summarization/run_summarization.py' \
    --model_name_or_path  '/kaggle/input/tesum-checkpoint-5-epochs/model_checkpoints/checkpoint-20370' \
    --do_train True\
    --do_eval True\
    --do_predict True\
    --train_file '/kaggle/input/tesum-dataset/TeSum_train_data.csv' \
    --validation_file '/kaggle/input/tesum-dataset/TeSum_dev_data.csv' \
    --test_file  '/kaggle/input/tesum-dataset/TeSum_test_data.csv' \
    --num_train_epochs 5 \
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
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
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
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Load the trained model and tokenizer
model_path = "/kaggle/input/tesum-checkpoint-5-epochs/model_checkpoints/checkpoint-20370/"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# Define a function for summarization
def generate_summary(article_text): #max_length=150
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

    /opt/conda/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



```python
# Example usage
article=""
summary = generate_summary(article)
print("Generated Summary:", summary)
```


```python

```
