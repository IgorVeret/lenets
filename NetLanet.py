import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Метод предварительной обработки изображения в переменную
transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
# После скачивания CIFAR10 и распаковки Меняем на  download=False. Скачивается в текущую папку.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Функция импортируется в первоначальное обучение, делится на порции,
# каждая партия по 4 фотографий для обучения batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
#Для расчета точности тестового набора
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Сеть Lanet
class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # print(nn.Conv2d())
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net=Lenet()
    # ---------После тренировки можно закомитить---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2): # Указываем количество проходов

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] Потери: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Обучение закончено')
    #---------------------------------------------------------------------------------

    PATH = './result.pt'    # Сохраняем результат обучения в текущую папку
    torch.save(net.state_dict(), PATH)   #После обучения можно закомитить
    net.load_state_dict(torch.load(PATH))    #Загружаем файлик обучения
    # подсчет прогнозов для каждого класса
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # Собрать правильные прогнозы для каждого класса
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Выводим результат на экран
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Точность для класса: {classname:5s} составляет {accuracy:.1f} %')
#----------------------------------------------------------------------------------
    # Оценить точность
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Рассчитать выходные данные, пропустив изображения через сеть
            outputs = net(images)
            # Выбор в качестве прогноза
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность сети на 10000 тестовых изображений: {100 * correct // total} %')
