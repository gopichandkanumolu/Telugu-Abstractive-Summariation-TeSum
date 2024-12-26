```python
import pandas as pd
import os
from tqdm import tqdm
!pip install absl-py nltk numpy six>=1.14 pythainlp pyonmttok jieba fugashi[unidic]
```


```python
language='telugu'
lang_code = "te"
actual_fp='/kaggle/input/tesum-dataset/TeSum_test_data.csv'
predictions_fp='/kaggle/input/tesum-checkpoint-5-epochs/model_checkpoints/generated_predictions.txt'
```


```python
df=pd.read_csv(actual_fp)
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>cleaned_text</th>
      <th>summary</th>
      <th>article_sentence_count</th>
      <th>article_token_count</th>
      <th>summary_token_count</th>
      <th>title_token_count</th>
      <th>compression</th>
      <th>abstractivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18081</td>
      <td>https://www.prajasakti.com/WEBSECTION/National...</td>
      <td>పోర్చుగీసు ప్రధానితో మోడీ భేటీ.\n</td>
      <td>భారత ప్రధాన మంత్రి నరేంద్ర మోడీ కొద్ది సేపటి క...</td>
      <td>భారత ప్రధాన మంత్రి నరేంద్ర మోడీ నేడు పోర్చుగీస...</td>
      <td>4</td>
      <td>36</td>
      <td>18</td>
      <td>6</td>
      <td>50.0</td>
      <td>16.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
actual_labels=list(df['summary'].values)

with open(predictions_fp) as fp:
    predicted_labels=fp.readlines()
    predicted_labels=[i.strip() for i in predicted_labels]

if len(actual_labels)==len(predicted_labels):
    print('Sizes Matched')
else:
    print('WARNING: Actual and Predicted Sizes Not Matched')
```

    Sizes Matched



```python
len(df), len(predicted_labels)
```




    (2017, 2017)



# Multilingual ROUGE


```python
os.chdir('/kaggle/input/multi-lingual-rouge-score/multilingual rouge')
from rouge_score import rouge_scorer
os.chdir('/kaggle/working')
```


```python
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, lang=language)
```


```python
def return_rouge_score(ref,hyp):
    scores = scorer.score(ref, hyp) # ref and hyp must be in string format only
    r1_f=scores['rouge1'][2] # index 2 corresponds to f-1 score, 0-precsion,1-recall
    r2_f=scores['rouge2'][2]
    rL_f=scores['rougeL'][2]
    
    return r1_f,r2_f,rL_f
```


```python
r1=[]
r2=[]
rL=[]

references=actual_labels
hypothesis=predicted_labels
```


```python
for i in range(len(actual_labels)):
#     r1_f,r2_f,rL_f=return_rouge_score(get_encoded(references[i]),get_encoded(hypothesis[i]))
    r1_f,r2_f,rL_f=return_rouge_score(str(references[i]),str(hypothesis[i]))
    
    r1.append(r1_f)
    r2.append(r2_f)
    rL.append(rL_f)
```


```python
df['Prediction']=predicted_labels
df['ROUGE-1']=r1
df['ROUGE-2']=r2
df['ROUGE-L']=rL
```


```python
df.sort_values(by=['ROUGE-L'],inplace=True,ascending=False)
```


```python
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>cleaned_text</th>
      <th>summary</th>
      <th>article_sentence_count</th>
      <th>article_token_count</th>
      <th>summary_token_count</th>
      <th>title_token_count</th>
      <th>compression</th>
      <th>abstractivity</th>
      <th>Prediction</th>
      <th>ROUGE-1</th>
      <th>ROUGE-2</th>
      <th>ROUGE-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>37579</td>
      <td>https://www.prajasakti.com/WEBSECTION/Internat...</td>
      <td>నూతన అదనపు భవనాల నిర్మాణ ప్రక్రియ.</td>
      <td>విజయనగరం జిల్లా బొబ్బిలి పట్టణంలోని మున్సిఫ్ మ...</td>
      <td>విజయనగరం జిల్లా బొబ్బిలి పట్టణంలోని మున్సిఫ్ మ...</td>
      <td>4</td>
      <td>95</td>
      <td>38</td>
      <td>6</td>
      <td>60.0</td>
      <td>10.53</td>
      <td>విజయనగరం జిల్లా బొబ్బిలి పట్టణంలోని మున్సిఫ్ మ...</td>
      <td>0.84507</td>
      <td>0.724638</td>
      <td>0.84507</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['ROUGE-1','ROUGE-2','ROUGE-L']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ROUGE-1</th>
      <th>ROUGE-2</th>
      <th>ROUGE-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2017.000000</td>
      <td>2017.000000</td>
      <td>2017.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.387859</td>
      <td>0.224620</td>
      <td>0.321154</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.129061</td>
      <td>0.120668</td>
      <td>0.121375</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.295652</td>
      <td>0.132231</td>
      <td>0.231579</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.387097</td>
      <td>0.215385</td>
      <td>0.315789</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.480769</td>
      <td>0.304000</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.845070</td>
      <td>0.731707</td>
      <td>0.845070</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
