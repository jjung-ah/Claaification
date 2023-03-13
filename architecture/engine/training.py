# Coding by BAEK(01153450@hyundai-autoever.com)

import os
import torch
import torch.nn as nn
# import torch.optim as optim
from architecture.solver.optimizer import Adam
from architecture.modeling.models import resnet
from tqdm import tqdm


class SupervisedLearning():

    def __init__(self, trainloader, valloader, model_name, pretrained):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.trainloader = trainloader
        self.valloader = valloader

        self.model_name = model_name
        self.model = resnet.modeltype(self.model_name)
        self.model = self.model.to(self.device)

        if pretrained != None:
            self.model.load.state_dict(torch.load(pretrained))
            print("Completed you pretrained model")

        print('Completed loading your networks')

        self.criterion = nn.CrossEntropyLoss()


    def val(self, dataloader):

        correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        return acc


    def train(self, epoch, lr, l2):
        # print(self.model.named_parameters())

        # optimizer = optim.Adam(self.model.named_parameters(), lr=lr, weight_decay=l2)  #230227
        optimizer = Adam(self.model, lr=lr, weight_decay=l2)

        train_loss_list = []
        val_loss_list = []
        n = len(self.trainloader)
        m = len(self.valloader)
        val_loss = 10  # dummy

        print('Start training the model')
        for epoch in tqdm(range(epoch)):

            running_loss = 0.0

            for data in self.trainloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                trainloss = self.criterion(outputs, labels)

                trainloss.backward()
                optimizer.step()

                running_loss += trainloss.item()

            train_cost = running_loss / n
            train_loss_list.append(train_cost)

            running_loss = 0.0

            for data in self.valloader:

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                valloss = self.criterion(outputs, labels)
                running_loss += valloss.item()

            val_cost = running_loss / m
            val_loss_list.append(val_cost)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_cost}, Validation Loss: {val_cost}')
                if epoch != 0:
                    torch.save(self.model.state_dict(),
                               os.path.join('/mnt/hdd1/datasets/hyundai-steel-goro/datasets/04_results/4th',
                                            f'{self.model_name}_{epoch}.pth'))

            if val_cost <= val_loss:
                # torch.save(self.model.state_dict(), './results/' + self.model_name + '_best.pth')
                torch.save(self.model.state_dict(),
                           os.path.join('/mnt/hdd1/datasets/hyundai-steel-goro/datasets/04_results/4th',
                                        f'{self.model_name}_{epoch}_best.pth'))
                val_loss = val_cost
                best_epoch = epoch

        # torch.save(self.model.state_dict(), './results/' + self.model_name + '_last.pth')
        torch.save(self.model.state_dict(),
                   os.path.join('/mnt/hdd1/datasets/hyundai-steel-goro/datasets/04_results/4th',
                                f'{self.model_name}_{epoch}_last.pth'))
        print('Finished Training')



        ###########################

        self.model.load_state_dict(torch.load(os.path.join('/mnt/hdd1/datasets/hyundai-steel-goro/datasets/04_results/4th',
                                        f'{self.model_name}_{epoch}_best.pth')))
        train_acc = SupervisedLearning.val(self.trainloader)
        val_acc = SupervisedLearning.val(self.valloader)
        print(f'Epoch {best_epoch}, Train Accuracy {train_acc}, Test Accuracy {val_acc}')


