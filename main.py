# -*-coding:utf-8-*-
import torch
from torch import nn
from d2l import torch as d2l
import BiRNN
import Init
import Embedding
import Predict
import Train

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
embed_size = 100
num_hiddens = 100
num_layers = 2
device = [0]
lr, num_epochs = 0.01, 10

if __name__ == '__main__':

    net = BiRNN.BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

    net.apply(Init.init_weights)

    glove_embedding = Embedding.TokenEmbedding('glove.6b.100d')

    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    Train.train(net, train_iter, test_iter, loss, optimizer, num_epochs, device)
    print(Predict.predict_sentiment(net, vocab, 'this movie is so great'))
    print(Predict.predict_sentiment(net, vocab, 'this movie is so bad'))
