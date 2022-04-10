# Make Inference pipeline with % of top 3 + combine Testing pipeline
# Nxt step : Multiple images Inference to get more accurate result (Avoid photo quality issue)

import torch
import torch.utils.data
from torchvision import datasets, transforms
from Densenet import get_model
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dir = r'C:\test\testing'
inference_dir = r'C:\test\Inference'

binary_dic = {0: 'Normal',
              1 : 'Illness detected'}

dic = {0: 'Blepharitis + Conjunctivitis',
       1: 'Cataract',
       2: 'Cherry Eye',
       3: 'Corneal Edema + Corneal Ulceration',
       4: 'Glaucoma',
       5: 'Hyphema + Uveitis',
       6: 'KCS',
       7: 'Nuclear Sclerosis'}


class Test_Infernce():
    def __init__(self, binary_model, model, test_dir, inference_dir, inference=True):
        self.test_dir = test_dir
        self.inference_dir = inference_dir
        self.inference = inference

        self.testloader = self.load_test_data()
        self.binary_model = binary_model
        self.model = model

    def load_test_data(self):
        transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(244),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

        if self.inference:
            test_data = datasets.ImageFolder(self.inference_dir, transform=transform)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=1)

        else:
            test_data = datasets.ImageFolder(self.test_dir, transform=transform)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=16)

        return testloader

    def Inference_test(self):
        with torch.no_grad():
            binary_model = self.binary_model.to(device)
            binary_model.eval()

            model = self.model.to(device)
            model.eval()

            if not self.inference:
                top3_res = []
                pred = []
                label = []

                for images, labels in self.testloader:
                    images, labels = images.to(device), labels.to(device)

                    output = model(images)
                    ps = torch.exp(output)
                    reshaped_labels = labels.reshape(-1, 1)

                    top3 = ps.topk(3, dim=1)[1]

                    for i in range(reshaped_labels.shape[0]):
                        if reshaped_labels[i] in top3[i]:
                            top3_res.append(1)
                        else:
                            top3_res.append(0)

                    pred.append(ps.max(dim=1)[1])
                    label.append(labels.data)

                pred = torch.cat(pred).tolist()
                label = torch.cat(label).tolist()

                confusion = confusion_matrix(label, pred)

                print('Top 3 prediction Accuracy: ', sum(top3_res) / len(top3_res))
                print(confusion)

            else:
                res_no = 0
                binary_res = None
                res = None

                for images, labels in self.testloader:
                    images = images.to(device)

                    binary_output = binary_model(images)
                    output = model(images)
                    ps = torch.exp(output)
                    if binary_res is None:
                        binary_res = binary_output
                    if res is None:
                        res = ps
                    else:
                        binary_res += binary_output
                        res += ps

                    res_no += 1

                binary_score = binary_res / res_no

                top3 = res.topk(3, dim=1)[1]
                top3_percent = res.topk(3, dim=1)[0] / res_no

                top3_percent_list = top3_percent.tolist()[0]

                print("The predicted results is:")
                if binary_score < 0.5:
                    print('1.Normal Eye : 90%')
                    for i in range(len(top3_percent_list)-1):
                        print(f'{i + 2}.{dic[top3[0][i].item()]}'
                              , '5%')

                else:
                    for i in range(len(top3_percent_list)):
                        # Map top 3 with class name
                        print(f'{i + 1}.{dic[top3[0][i].item()]}'
                              , f', {top3_percent_list[i] / sum(top3_percent_list)}')

if __name__ == '__main__':
    a = Test_Infernce(get_model(), get_model(pretrained='multi'), test_dir, inference_dir)
    a.Inference_test()
