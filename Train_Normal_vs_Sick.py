import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models

from PIL import Image
from sklearn.metrics import confusion_matrix

import time
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = r'C:\binary model\train'
valid_tir = r'C:\binary model\valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_tir, transform=valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=16)


class NN_Classifier_binary(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.1):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output(x)

        return F.sigmoid(x).squeeze(1)


def validation(model, testloader, criterion):

    pred = []
    label = []

    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output.type(torch.FloatTensor), labels.type(torch.FloatTensor)).item()

        equality = (labels.data.cpu().detach().numpy()
                    == [i.cpu().detach().numpy().round().astype('int64') for i in output])

        equality = torch.tensor(equality)
        accuracy += equality.type(torch.FloatTensor).mean()

        pred.append(output)
        label.append(labels.data)

    pred = torch.cat(pred).tolist()
    pred = [round(i) for i in pred]
    label = torch.cat(label).tolist()

    confusion = confusion_matrix(label, pred)

    return test_loss, accuracy, confusion

def make_NN(n_hidden=[1024, 512, 256], n_drop=0.1, n_epoch=2, labelsdict=1, lr=1e-3, device=device, model_name='densenet201'):  # resnet50 vgg16 densenet169
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
    model.classifier = NN_Classifier_binary(input_size=n_in, output_size=n_out
                                     , hidden_layers=n_hidden, drop_p=n_drop)

    # Define criterion and optimizer
    criterion = nn.BCELoss()
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

            output = model(images)
            loss = criterion(output.type(torch.FloatTensor), labels.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        # Eval mode for predictions
        model.eval()

        # Turn off gradients for validation
        with torch.no_grad():
            test_loss, accuracy, confusion = validation(model, validloader, criterion)

        print("Epoch: {}/{} - ".format(e + 1, epochs),
              "Training Loss: {:.3f} - ".format(running_loss / steps+1),
              "Validation Loss: {:.3f} - ".format(test_loss / len(validloader)),
              "\nValidation Accuracy: {:.3f} -".format(accuracy / len(validloader)),
              "lr: {}".format(optimizer.param_groups[0]['lr']))


        print(f'\n{confusion}')

        if model_loss == 0:
            model_loss += test_loss
            model_acc += accuracy
            torch.save(model.state_dict(), f'binary_model_{str(e)}.pt')
        else:
            if (model_loss >= test_loss) and (model_acc <= accuracy):
                model_loss = test_loss
                model_acc = accuracy
                torch.save(model.state_dict(), f'binary_model_{str(e)}.pt')


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
    model_binary = make_NN(n_epoch=50)








