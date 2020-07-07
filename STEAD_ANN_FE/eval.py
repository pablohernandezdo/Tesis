import time
import tqdm
import torch
import argparse

from dataset import HDF5Dataset
from torch.utils.data import DataLoader

from model import Classifier, Classifier_XS, Classifier_S, Classifier_XL, Classifier_XXL, Classifier_XXXL


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='XXL_lr0000001_bs32', help="Name of model to eval")
    parser.add_argument("--classifier", default='XXL', help="Choose classifier architecture, C, S, XS, XL, XXL, XXXL")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    args = parser.parse_args()

    print(f'Evaluation details: \n {args}\n')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = HDF5Dataset('../' + args.train_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = HDF5Dataset('../' + args.test_path)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

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
    net.load_state_dict(torch.load('../models/' + args.model_name + '.pth'))
    net.eval()

    # Evaluate on training set

    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    with tqdm.tqdm(total=len(trainloader), desc='Train dataset evaluation', position=0) as train_bar:
        with torch.no_grad():
            for data in trainloader:
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
                train_bar.update(1)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

    eval_1 = time.time()

    print(f'Training evaluation time: {eval_1 - start_time}\n'
          f'Evaluation results: \n'
          f'correct: {correct}, total: {total}\n'
          f'True positives: {tp}\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'F-score: {fscore:5.3f}\n')

    print('Accuracy of the network on the train set: %d %%\n' % (100 * correct / total))

    # Evaluate on test set

    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    with tqdm.tqdm(total=len(testloader), desc='Test dataset evaluation', position=0) as test_bar:
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
                test_bar.update(1)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

    print(f'Test evaluation time: {time.time() - eval_1}\n'
          f'Evaluation results: \n'
          f'correct: {correct}, total: {total}\n'
          f'True positives: {tp}\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'F-score: {fscore:5.3f}\n')

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()
