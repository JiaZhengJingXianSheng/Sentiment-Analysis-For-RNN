# Sentiment-Analysis-For-RNN
# 循环神经网络进行情感分析

## **引言：**

对于情感分析，如果简化来看可以分为**正向情绪**和**负向情绪**，我们可以将情感分析视为文本分类任务，因此我们可以将预训练的词向量应用于情感分析。我们可以用预训练的GloVe模型表示每个标记，并反馈到RNN中。

![I8V7i6.png](https://z3.ax1x.com/2021/11/08/I8V7i6.png)

## RNN表征文本

在文本分类任务中，要将可变长度的文本序列转为固定长度。可以通过**nn.Embedding()**函数获得单独的预训练GloVe，再去通过**双向LSTM**，最后在去通过一个**全连接层**做一个二分类，即可实现RNN表征文本。

```python
self.embedding = nn.Embedding(vocab_size, embed_size)
self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,bidirectional=True)
self.decoder = nn.Linear(4 * num_hiddens, 2)
```

```python
embeddings = self.embedding(inputs.T)
outputs, _ = self.encoder(embeddings)
encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
outs = self.decoder(encoding)
```

## 训练和评估模型

选用**IMDB数据集**，该数据集包含**50 000 条严重两极分化的评论。 训练集 测试集各 25000 条评论，并都包含 50% 的正面评论和 50% 的负面评论。**加载我们预训练的GloVe模型后调用训练函数进行训练，用100维GloVe下方展示模型情况。

```
  (embedding): Embedding(49346, 100)
  (encoder): LSTM(100, 100, num_layers=2, bidirectional=True)
  (decoder): Linear(in_features=400, out_features=2, bias=True)

```

### **训练情况如下**

![I8KfsO.png](https://z3.ax1x.com/2021/11/08/I8KfsO.png)

用训练好的模型预测两个简单的句子

```python
predict_sentiment(net, vocab, 'this movie is so great')
```

'positive'

```
predict_sentiment(net, vocab, 'this movie is so bad')
```

'negative'



# 参考链接

https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html
