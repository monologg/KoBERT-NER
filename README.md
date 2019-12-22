# KoBERT-NER

- NER Task with KoBERT
- Implement with `Huggingface Tranformers` library

## Dependencies

- torch>=1.1.0
- transformers>=2.2.2
- seqeval>=0.0.12
- sentencepiece>=0.1.82

## Dataset

- NER Dataset from **Naver NLP Challenge 2018** ([Github link](https://github.com/naver/nlp-challenge))
- Because there is only train dataset, test dataset was splitted from train dataset ([Data link](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging/data))
  - Train (81,000) / Test (9,000)

## How to use KoBERT on Huggingface Transformers Library

- From transformers v2.2.2, you can upload/download personal bert model directly.
- To use tokenizer, you have to import `KoBertTokenizer` from `tokenization_kobert.py`.

```python
from transformers import BertModel
from tokenization_kobert import KoBertTokenizer

model = BertModel.from_pretrained('monologg/kobert')
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
```

## Usage

```bash
$ python3 main.py --model_type kobert --do_train --do_eval
```

## Results

|                   | Slot F1 (%) |
| ----------------- | ----------- |
| KoBERT            | 84.23       |
| DistilKoBERT      | 81.22       |
| Bert-Multilingual | TBD         |
| BiLSTM-CRF        | 76.45       |

## References

- [Naver NLP Challenge](https://github.com/naver/nlp-challenge)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NLP Implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [BERT NER by eagle705](https://github.com/eagle705/pytorch-bert-crf-ner)
