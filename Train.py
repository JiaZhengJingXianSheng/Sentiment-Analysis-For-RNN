import matplotlib.pyplot as plt
import d2l.torch as d2l
from torch import nn
import numpy as np


def train(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    val_acc_list = []
    lossList = []
    train_acc_list = []
    x = np.linspace(0, (num_epochs - 1), num_epochs)
    for epoch in range(num_epochs):
        running_train_acc = 0.0
        train_acc_End = 0.0
        running_loss = 0.0
        lossEnd = 0.0
        for i, (features, labels) in enumerate(train_iter):
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)

            rate = (i + 1) / len(train_iter)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            running_loss += (l.item() / labels.shape[0])
            lossEnd = (running_loss / (i + 1))
            running_train_acc += (acc / labels.numel())
            train_acc_End = running_train_acc / (i + 1)
            print(
                "\rEpoch: {}  {:^3.0f}%[{}->{}] train loss: {:.5f}\ttrain acc: {:.5f}".format(epoch, int(rate * 100), a,
                                                                                              b, lossEnd,
                                                                                              train_acc_End), end="")
        lossList.append(lossEnd)
        train_acc_list.append(train_acc_End)
        print()
        # val
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        val_acc_list.append(test_acc)
        print('Epoch: {}   val_accuracy: {:.5f}'.format(epoch, test_acc))
    print('Finished Training')

    plt.plot(x, lossList, 'or-', label=r'train_loss')
    plt.plot(x, train_acc_list, 'ob-', label=r'train_accuracy')
    plt.plot(x, val_acc_list, 'og-', label=r'val_accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('End.png')
    plt.show()
