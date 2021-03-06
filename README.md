# naivenlp

![Python package](https://github.com/luozhouyang/naivenlp/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/naivenlp.svg)](https://badge.fury.io/py/naivenlp)
[![Python](https://img.shields.io/pypi/pyversions/naivenlp.svg?style=plastic)](https://badge.fury.io/py/naivenlp)


NLP常用工具包。



主要包含以下模块：

- [naivenlp](#naivenlp)
  - [Installation](#installation)
  - [Tokenizers](#tokenizers)
    - [JiebaTokenizer的使用](#jiebatokenizer的使用)
    - [CustomTokenizer的使用](#customtokenizer的使用)
    - [BasicTokenizer的使用](#basictokenizer的使用)
    - [WordpieceTokenizer的使用](#wordpiecetokenizer的使用)
    - [TransformerTokenizer的使用](#transformertokenizer的使用)
    - [BertTokenizer的使用](#berttokenizer的使用)
  - [Correctors](#correctors)
    - [n-gram语言模型和词典纠错](#n-gram语言模型和词典纠错)
    - [基于深度学习的纠错](#基于深度学习的纠错)
  - [Similarity](#similarity)
  - [Structures](#structures)
    - [Trie的使用](#trie的使用)
  - [Utils](#utils)
  - [Datasource](#datasource)
    - [下载搜狗词库](#下载搜狗词库)


## Installation

```bash
pip install -U naivenlp
```

Install extra dependencies:

```bash
pip install pycorrector
pip install git+https://github.com/kpu/kenlm.git
```


## Tokenizers

`Tokenizer`的作用是**分词**， 同时具有把词语映射到ID的功能。

`naivenlp.tokenizers`模块包含以下`Tokenizer`实现：

* `JiebaTokenizer`，继承自`VocabBasedTokenizer`，分词使用`jieba`
* `CustomTokenizer`，继承自`VocabBasedTokenizer`，基于词典文件的`Tokenizer`，包装`tokenize_fn`自定义函数来实现各种自定义的`Tokenizer`
* `TransformerTokenizer`，继承自`VocabBasedTokenizer`，用于`Transformer`模型分词
* `BertTokenizer`，继承自`VocabBasedTokenizer`，用于`BERT`模型分词



### JiebaTokenizer的使用

分词过程使用`jieba`。

```python
from naivenlp.tokenizers import JiebaTokenizer

tokenizer = JiebaTokenizer(
    vocab_file='vocab.txt',
    pad_token='[PAD]',
    unk_token='[UNK]',
    bos_token='[BOS]',
    eos_token='[EOS]',
)

tokenizer.tokenize('hello world!', mode=0, hmm=True)

tokenizer.encode('hello world!', add_bos=False, add_eos=False)

```

### CustomTokenizer的使用

方便用户自定义分词过程。

以使用`baidu/lac`来分词为例。

```bash
pip install lac
```

```python
from naivenlp.tokenizers import CustomTokenizer

from LAC import LAC

lac = LAC(mode='seg')

def lac_tokenize(text, **kwargs):
    return lac.run(text)


tokenizer = CustomTokenizer(
    vocab_file='vocab.txt',
    tokenize_fn=lac_tokenize,
    pad_token='[PAD]',
    unk_token='[UNK]',
    bos_token='[BOS]',
    eos_token='[EOS]',
)

tokenizer.tokenize('hello world!')

tokenizer.encode('hello world!', add_bos=False, add_eos=False)

```

### BasicTokenizer的使用

这个分词器的使用很简单。不需要词典。它会根据空格来分词。它有以下功能：

* 按照空格和特殊字符分词
* 根据设置，决定是否**大小写转换**
* 根据设置，切分汉字，按照字的粒度分词

```python
from naivenlp.tokenizers import BasicTokenizer

tokenizer = BasicTokenizer(do_lower_case=True, tokenize_chinese_chars=True)

tokenizer.tokenize('hello world, 你好世界')

```


### WordpieceTokenizer的使用

`Wordpiece`是一种分词算法，具体请自己查询相关文档。

`WordpieceTokenizer`需要传入一个词典map。

```python
from naivenlp.tokenizers import WordpieceTokenizer

tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='[UNK]')

tokenizer.tokenize('hello world, 你好世界')
```


### TransformerTokenizer的使用

```python
from naivenlp.tokenizers import TransformerTokenizer


tokenizer = TransformerTokenizer(vocab_file='vocab.txt')

tokenizer.tokenize('Hello World, 你好世界')

tokenizer.encode('Hello World, 你好世界', add_bos=False, add_eos=False)

```

### BertTokenizer的使用

```python
from naivenlp.tokenizers import BertTokenizer


tokenizer = BertTokenizer(vocab_file='vocab.txt', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')

tokenizer.tokenize('Hello World, 你好世界')

tokenizer.encode('Hello World, 你好世界', add_bos=False, add_eos=False)

```


## Correctors

文本纠错，包括传统的n-gram语言模型和词典的方式，也可以使用基于深度学习的方法。

### n-gram语言模型和词典纠错

这里的`KenLMCorrector`是对 [shibing624/pycorrector](https://github.com/shibing624/pycorrector) 项目的包装。

```python
from naivenlp.correctors import KenLMCorrector

c = KenLMCorrector()
texts = [
    '软件开发工成师',
    '少先队员因该为老人让坐',
]

for text in texts:
    print(c.correct(text))

```
可以得到纠错结果：

```bash
('软件开发工程师', [('工成师', '工程师', 4, 7)])
('少先队员应该为老人让座', [('因该', '应该', 4, 6), ('坐', '座', 10, 11)])
```

### 基于深度学习的纠错

主要是利用`seq2seq`模型完成纠错。例如：

* `RNN` + `Attention` 传统的`seq2seq` 模型
* `Transformer`模型

模型的训练使用 [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) 库，训练方法请到此项目的文档查看。

以下是使用`Transformer`模型的示例：

```bash
>>> from naivenlp.correctors import TransformerCorrector
>>> SAVED_MODEL='/models/correction_models/transformer-step-2000' # 此处换成你自己训练的模型，SavedModel格式
>>> corrector = TransformerCorrector(SAVED_MODEL)
>>> result, prob = corrector.correct('我最近每天晚上都会拧着鼻子去喝30cc的醋了。')
>>> print('result: ', result)
result:  我最近每天晚上都会拧着鼻子去喝30cc的醋。
>>> print('  prob: ', prob)
  prob:  -6.088574
>>>
```

## Similarity

多种字符串相似度的度量。是对[luozhouyang/python-string-similarity](https://github.com/luozhouyang/python-string-similarity)的包装。

```bash
>>> import naivenlp
>>> a = 'ACCTTTDEX'
>>> b = 'CGGTTEEXX'
>>> naivenlp.cosine_distance(a, b)
1.0
>>> naivenlp.cosine_similarity(a, b)
1.0
>>> naivenlp.jaccard_distance(a, b)
1.0
>>> naivenlp.jaccard_similarity(a, b)
0.0
>>> naivenlp.levenshtein_distance(a, b)
5
>>> naivenlp.levenshtein_distance_normalized(a, b)
0.5555555555555556
>>> naivenlp.levenshtein_similarity(a, b)
0.4444444444444444
>>> naivenlp.weighted_levenshtein_distance(a, b)
5.0
>>> naivenlp.damerau_distance(a, b)
5
>>> naivenlp.lcs_distance(a, b)
8
>>> naivenlp.lcs_length(a, b)
5
>>> naivenlp.sorense_dice_distance(a, b)
1.0
>>> naivenlp.sorense_dice_similarity(a, b)
0.0
>>> naivenlp.optimal_string_alignment_distance(a, b)
5
>>> 
```

## Structures

常用的数据结构实现。

目前支持：

* 字典树Trie


### Trie的使用

```bash
>>> import naivenlp
>>> trie = naivenlp.Trie()
>>> trie.put('上海市浦东新区')
>>> trie.show()
.
+----上
|    +----海
|    |    +----市
|    |    |    +----浦
|    |    |    |    +----东
|    |    |    |    |    +----新
|    |    |    |    |    |    +----区
>>> trie.put('上海市黄浦区')
>>> trie.show()
.
+----上
|    +----海
|    |    +----市
|    |    |    +----浦
|    |    |    |    +----东
|    |    |    |    |    +----新
|    |    |    |    |    |    +----区
|    |    |    +----黄
|    |    |    |    +----浦
|    |    |    |    |    +----区
>>> 
>>> for r in trie.keys_with_prefix('上海市'):
...     print(r)
... 
['上', '海', '市', '浦', '东', '新', '区']
['上', '海', '市', '黄', '浦', '区']
>>> 

```

## Utils

常用文本操作：

* `naivenlp.q2b(s)` 全角转半角
* `naivenlp.b2q(s)` 半角转全角
* `naivenlp.split_sentence(s)` 把长文本切分成句子列表


## Datasource

数据收集模块。目前支持：

* 下载所有的搜狗词库保存成文本文件


### 下载搜狗词库

```python
from naivenlp.datasources import sogou as sg

# 下载category_id=1下面所有的词典，保存到/tmp/sogou
sg.download_category(1, '/tmp/sogou')

# 下载所有category保存到/tmp/sogou
sg.download_all_category('/tmp/sogou')

# 把下载的所有文件合成一个文件
sg.collect('/tmp/sogou', './sogou.vocab', maxlen=6)

```
