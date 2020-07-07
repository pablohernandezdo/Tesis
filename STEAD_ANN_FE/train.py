import time
import tqdm
import argparse

import torch
import torch.optim as optim
import torch.nn as nn

from .dataset import HDF5Dataset
from torch.utils.data import DataLoader

from .model import Classifier

from torch.utils.tensorboard import SummaryWriter


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to save")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay parameter")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    args = parser.parse_args()

    print(f'Execution details: \n {args}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tb = SummaryWriter('../runs/Seismic')

    train_dataset = HDF5Dataset('../' + args.train_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = HDF5Dataset('../' + args.test_path)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    net = Classifier()
    net.to(device)

    traces, labels = next(iter(trainloader))
    traces, labels = traces.to(device), labels.to(device)
    tb.add_graph(net, traces)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    loss_id = 0

    # epoch_bar = tqdm.tqdm(total=args.n_epochs, desc='Epochs', position=0)

    with tqdm.tqdm(total=args.n_epochs, desc='Epochs', position=0) as epoch_bar:

        for epoch in range(args.n_epochs):

            total_loss = 0
            # batch_bar = tqdm.tqdm(total=len(trainloader), desc='Batches', position=1)

            with tqdm.tqdm(total=len(trainloader), desc='Batches', position=1) as batch_bar:

                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    tb.add_scalar('Output', outputs[0].item(), loss_id)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    loss_id += 1

                    tb.add_scalar('Loss', loss.item(), loss_id)
                    batch_bar.update(1)

                tb.add_scalar('Total_Loss', total_loss, epoch)
                epoch_bar.update(1)

    tb.close()
    torch.save(net.state_dict(), '../models/' + args.model_name + '.pth')
    training_time = time.time()

    net.eval()

    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    with torch.no_grad():
        for data in testloader:
            traces, labels = data[0].to(device), data[1].to(device)
            outputs = net(traces)
            predicted = torch.round(outputs.data)
            total += labels.size(0)

            for i, pred in enumerate(predicted):
                if pred:
                    if pred == labels[i]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if pred == labels[i]:
                        tn += 1
                    else:
                        fn += 1

            correct += (predicted == labels).sum().item()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

    print(f'\nTraining time: {training_time - start_time}\n'
          f'Evaluation time: {time.time() - training_time}\n'
          f'Execution time: {time.time() - start_time}\n\n'
          f'Evaluation results:\n'
          f'correct: {correct}, total: {total}\n'
          f'True positives: {tp}\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Precision: {precision:5.2f}\n'
          f'Recall: {recall:5.2f}\n'
          f'F-score: {fscore:5.2f}\n')

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()
