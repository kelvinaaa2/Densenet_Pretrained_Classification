import json
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
# https://github.com/MiguelAMartinez/flowers-image-classifier/blob/master/image_classifier.ipynb
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings('ignore')


# Define dir paths
train_dir = r'C:\test\train'
valid_dir = r'C:\test\valid'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your transforms for the training, validation, and testing sets

# Add random transforms in training set for better generalization
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=16)


# Define classifier class
class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


# Define validation function
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    accuracy_2 = 0
    accuracy_3 = 0

    pred = []
    label = []

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)

        # Count top 3 outputs as correct
        top2 = ps.topk(2, dim=1)[1]
        top3 = ps.topk(3, dim=1)[1]

        reshape_labels = labels.data.view(-1,1)

        top2_res = []
        for i in range(reshape_labels.shape[0]):
            if reshape_labels[i] in top2[i]:
                top2_res.append(1)
            else:
                top2_res.append(0)

        top2_res = torch.tensor(top2_res)
        accuracy_2 += top2_res.type(torch.FloatTensor).mean()

        top3_res = []
        for i in range(reshape_labels.shape[0]):
            if reshape_labels[i] in top3[i]:
                top3_res.append(1)
            else:
                top3_res.append(0)

        top3_res = torch.tensor(top3_res)
        accuracy_3 += top3_res.type(torch.FloatTensor).mean()

        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        pred.append(ps.max(dim=1)[1])
        label.append(labels.data)

    pred = torch.cat(pred).tolist()
    label = torch.cat(label).tolist()

    confusion = confusion_matrix(label, pred)

    return test_loss, accuracy, accuracy_2, accuracy_3, confusion

def confusion(model, testloader, device):
    pass



# Define NN function # [512, 256, 128]
def make_NN(n_hidden=[1024, 512, 256], n_drop=0.1, n_epoch=2, labelsdict=8, lr=1e-3, device=device, model_name='densenet201'):  # resnet50 vgg16 densenet169
    # Import pre-trained NN model
    model = getattr(models, model_name)(pretrained=True)

    # Freeze parameters that we don't need to re-train
    for param in model.features[:-3].parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Make classifier
    n_in = next(model.classifier.modules()).in_features #Densenet
    # n_in = model.fc.in_features #Resnet50
    n_out = labelsdict
    model.classifier = NN_Classifier(input_size=n_in, output_size=n_out
                                     , hidden_layers=n_hidden, drop_p=n_drop)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam([{'params': model.features[-3:].parameters()}
                               , {'params' : model.classifier.parameters()}], lr=lr)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11, 30])  # [8, 20, 25]

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0
    running_loss = 0

    # For plotting loss
    epoch_step = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    model_loss = 0
    model_acc = 0

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()


        # Eval mode for predictions
        model.eval()

        # Turn off gradients for validation
        with torch.no_grad():
            test_loss, accuracy, accuracy_2, accuracy_3, confusion = validation(model, validloader, criterion, device)

        print("Epoch: {}/{} - ".format(e + 1, epochs),
              "Training Loss: {:.3f} - ".format(running_loss / steps+1),
              "Validation Loss: {:.3f} - ".format(test_loss / len(validloader)),
              "\nValidation Accuracy: {:.3f} -".format(accuracy / len(validloader)),
              "\nValidation Accuracy Top2: {:.3f} -".format(accuracy_2 / len(validloader)),
              "\nValidation Accuracy Top3: {:.3f} -".format(accuracy_3 / len(validloader)),
              "lr: {}".format(optimizer.param_groups[0]['lr']))


        print(f'\n{confusion}')

        if model_loss == 0:
            model_loss += test_loss
            model_acc += accuracy
            torch.save(model.state_dict(), f'model_{str(e)}.pt')
        else:
            if (model_loss >= test_loss) and (model_acc <= accuracy):
                model_loss = test_loss
                model_acc = accuracy
                torch.save(model.state_dict(), f'model_{str(e)}.pt')


        epoch_step.append(str(e))
        train_loss_history.append(running_loss / steps+1)
        val_loss_history.append(test_loss / len(validloader))
        val_acc_history.append(accuracy / len(validloader))

        running_loss = 0

        # Make sure training is back on
        model.train()

        lr_scheduler.step()



    # Add model info
    model.classifier.n_in = n_in
    model.classifier.n_hidden = n_hidden
    model.classifier.n_out = n_out
    model.classifier.lr = lr
    model.classifier.optimizer_state_dict = optimizer.state_dict
    model.classifier.model_name = model_name
    model.classifier.class_to_idx = train_data.class_to_idx

    print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start) / 60:.3f} min")

    plt.plot(epoch_step, train_loss_history, 'g', label='Training Loss')
    plt.plot(epoch_step, val_loss_history, 'b', label='Valid Loss')
    plt.plot(epoch_step, val_acc_history, 'r', label='Valid Loss')
    plt.title('Loss & Accuracy Curve')
    plt.xlabel('Epoch + Steps')
    plt.ylabel('Loss / Accuracy')
    plt.show()

    return model, epoch_step, train_loss_history, val_loss_history, val_acc_history

if __name__ == '__main__':
    model_a = make_NN(n_epoch=50)




