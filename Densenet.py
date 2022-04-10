import torch
from torchvision import models
from pets import NN_Classifier
from Train_Normal_vs_Sick import NN_Classifier_binary


def get_model(pretrained='binary'):
    # mode = 'binary' / 'multi'

    binary_model = r'binary_model_10.pt'
    multi_model = r'unfreeze4_X_0750.pt'

    model = models.densenet201(pretrained=False)

    n_in = next(model.classifier.modules()).in_features
    n_hidden = [1024, 512, 256]

    if pretrained == 'multi':
        n_out = 8  # 8 classes
        model.classifier = NN_Classifier(input_size=n_in, output_size=n_out,
                                         hidden_layers=n_hidden, drop_p=0)

        model.load_state_dict(torch.load(multi_model))

    else:
        assert (pretrained == 'binary'), "mode can only be 'multi' or 'binary'"
        n_out = 1
        model.classifier = NN_Classifier_binary(input_size=n_in, output_size=n_out,
                                         hidden_layers=n_hidden, drop_p=0)

        model.load_state_dict(torch.load(binary_model))

    return model



