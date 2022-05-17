# ------------------------------------------------------------------------------
#  HyperNOMAD - Hyper-parameter optimization of deep neural networks with
#		        NOMAD.
#
#
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
#  for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#  You can find information on the NOMAD software at www.gerad.ca/nomad
# ------------------------------------------------------------------------------

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
from datahandler2 import *
import numpy as np


# sys.path.append(os.environ.get('HYPERNOMAD_HOME')+"/src/blackbox/blackbox")


class Evaluator(object):
    def __init__(self, device, cnn, trainloader, testloader, optimizer, batch_size, dataset):
        self.__device = device
        self.__cnn = cnn
        self.__trainloader = trainloader
        self.__testloader = testloader
        self.__batch_size = batch_size
        self.__optimizer = optimizer
        self.__dataset = dataset
        self.__train_acc = None
        self.__val_acc = None
        self.__test_acc = None
        self.__best_epoch = None

    @property
    def device(self):
        return self.__device

    @property
    def cnn(self):
        return self.__cnn

    @cnn.setter
    def cnn(self, new_cnn):
        self.__cnn = new_cnn

    @property
    def trainloader(self):
        return self.__trainloader

    @property
    def testloader(self):
        return self.__testloader

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def dataset(self):
        return self.__dataset

    def train(self):
        criterion = nn.CrossEntropyLoss().cuda()

        if torch.cuda.is_available():
            self.cnn = torch.nn.DataParallel(self.cnn)
            cudnn.benchmark = True

        epoch = 0
        stop = False
        l_val_acc = []
        l_train_acc = []
        best_val_acc = 0
        epochs = []
        max_epochs = 300
        if self.dataset == 'MINIMNIST':
            max_epochs = 50

        # LR scheduler - SGD only
        # T_max = max_epochs // 3
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, eta_min=7.5e-4)

        while (not stop) and (epoch < max_epochs):
            self.cnn.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.cnn(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                self.__train_acc = 100. * correct / total

            l_train_acc.append(self.__train_acc)

            self.cnn.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.testloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.cnn(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    self.__val_acc = 100. * val_correct / val_total
            if self.__val_acc > best_val_acc:
                best_val_acc = self.__val_acc
                torch.save(self.cnn.state_dict(), 'best_model.pth')
            l_val_acc.append(self.__val_acc)

            epochs.append(epoch)

            if self.optimizer.__class__.__name__ == 'SGD':
                if (epoch > 0) and (epoch % 100 == 0):
                    self.cnn.load_state_dict(torch.load('best_model.pth'))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10

            print("Epoch {},  Train accuracy: {:.3f}, Val accuracy: {:.3f}".format(epoch + 1, self.__train_acc,
                                                                                   self.__val_acc))
            epoch += 1

        print('> Finished Training')

        # get the best validation accuracy and the corresponding epoch
        best_epoch = np.argmax(l_val_acc)
        best_val_acc = l_val_acc[best_epoch]

        # use the saved net to assess the test accuracy
        print('Best validation accuracy and corresponding epoch number : {:.3f}/{}'.format(
            best_val_acc, best_epoch + 1))
        return best_val_acc, best_epoch, epoch

    # def test(self):
    #     criterion = nn.CrossEntropyLoss()
    #     self.cnn.load_state_dict(torch.load('best_model.pth'))
    #
    #     total_test = 0
    #     correct_test = 0
    #     self.cnn.eval()
    #     test_loss = 0
    #
    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets) in enumerate(self.testloader):
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             outputs = self.cnn(inputs)
    #             loss = criterion(outputs, targets)
    #             test_loss += loss.item()
    #             _, predicted = outputs.max(1)
    #             total_test += targets.size(0)
    #             correct_test += predicted.eq(targets).sum().item()
    #             test_acc = 100. * correct_test / total_test
    #
    #     # exit(0)
    #     return test_acc
