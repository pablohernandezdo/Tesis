import time
import tqdm
import argparse

from humanfriendly import format_timespan

import torch
import torch.nn as nn
import torch.optim as optim


from dataset import HDF5Dataset
from torch.utils.data import DataLoader

from model import *

from torch.utils.tensorboard import SummaryWriter


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to save")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, S, XS, XL, XXL, XXXL")
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

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start tensorboard SummaryWriter
    tb = SummaryWriter('../runs/Seismic')

    # Train dataset
    train_dataset = HDF5Dataset(args.train_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Test dataset
    test_dataset = HDF5Dataset(args.test_path)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    if args.classifier == 'XS':
        net = Classifier_XS()
    elif args.classifier == 'S':
        net = Classifier_S()
    elif args.classifier == 'XL':
        net = Classifier_XL()
    elif args.classifier == 'XXL':
        net = Classifier_XXL()
    elif args.classifier == 'XXXL':
        net = Classifier_XXXL()
    else:
        net = Classifier()
    net.to(device)

    # Add model graph to tensorboard
    traces, labels = next(iter(trainloader))
    traces, labels = traces.to(device), labels.to(device)
    tb.add_graph(net, traces)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    # Loss id for tensorboard logs
    loss_id = 0

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs', position=0) as epoch_bar:
        for epoch in range(args.n_epochs):

            total_loss = 0

            with tqdm.tqdm(total=len(trainloader), desc='Batches', position=1) as batch_bar:
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    tb.add_scalar('Output', outputs[0].item(), loss_id)
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    loss_id += 1

                    tb.add_scalar('Loss', loss.item(), loss_id)
                    batch_bar.update(1)

                tb.add_scalar('Total_Loss', total_loss, epoch)
                epoch_bar.update(1)

    # Close tensorboard
    tb.close()

    # Save model
    torch.save(net.state_dict(), '../models/' + args.model_name + '.pth')
    training_time = time.time()

    # Evaluate model on test dataset
    net.eval()

    # True/False Positives/Negatives
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

    # Measure training, evaluation and execution times
    end_tm = time.time()

    tr_t = training_time - start_time
    ev_t = end_tm - training_time
    ex_t = end_tm - start_time

    print(f'\nTraining time: {format_timespan(tr_t)}\n'
          f'Evaluation time: {format_timespan(ev_t)}\n'
          f'Execution time: {format_timespan(ex_t)}\n\n'
          f'Evaluation results:\n'
          f'correct: {correct}, total: {total}\n'
          f'True positives: {tp}\n\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Evaluation metrics:\n\n'
          f'Precision: {precision:5.2f}\n'
          f'Recall: {recall:5.2f}\n'
          f'F-score: {fscore:5.2f}\n')

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()
