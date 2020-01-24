# KoBERT-NER

- KoBERT를 이용한 한국어 Named Entity Recognition Task
- `Huggingface Tranformers` 라이브러리를 이용하여 구현

## Dependencies

- torch>=1.1.0
- transformers>=2.2.2
- seqeval>=0.0.12
- sentencepiece>=0.1.82

## Dataset

- **Naver NLP Challenge 2018**의 NER Dataset 사용 ([Github link](https://github.com/naver/nlp-challenge))
- 해당 데이터셋에 Train dataset만 존재하기에, Test dataset은 Train dataset에서 split하였습니다. ([Data link](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging/data))
  - Train (81,000) / Test (9,000)

## How to use KoBERT on Huggingface Transformers Library

- 기존의 KoBERT를 transformers 라이브러리에서 곧바로 사용할 수 있도록 맞췄습니다.
  - transformers v2.2.2부터 개인이 만든 모델을 transformers를 통해 직접 업로드/다운로드하여 사용할 수 있습니다
- Tokenizer를 사용하려면 `tokenization_kobert.py`에서 `KoBertTokenizer`를 임포트해야 합니다.

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
| DistilKoBERT      | 82.14       |
| Bert-Multilingual | 81.78       |
| BiLSTM-CRF        | 76.45       |

## References

- [Naver NLP Challenge](https://github.com/naver/nlp-challenge)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NLP Implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [BERT NER by eagle705](https://github.com/eagle705/pytorch-bert-crf-ner)
