import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

torch.manual_seed(1)

batch_size = 100
num_classes = 10
epochs = 20

import tensorflow as tf

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 1, 28, 28).astype('float32')
x_test = x_test.reshape(10000, 1, 28, 28).astype('float32')

# 正規化
x_train /= 255
x_test /= 255

y_train = y_train.astype('long')
y_test = y_test.astype('long')

ds_train = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
ds_test  = data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

dataloader_train = data.DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)

dataloader_test = data.DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=False)

class CNNModel (nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)

        x = x.view(-1, 64*14*14)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.relu(self.fc2(x))

model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

global_step = 0

def train(epoch, writer):
    model.train()
    steps = len(ds_train)//batch_size
    for step, (images, labels) in enumerate(dataloader_train, 1):
        global global_step
        global_step += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(1)
        if step % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, epochs, step, steps, loss.item()))
            writer.add_scalar('train/train_loss', loss.item() , global_step)