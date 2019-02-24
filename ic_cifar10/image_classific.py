import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import torch.nn as nn

train_data = torchvision.datasets.CIFAR10(root='./datasets',
	train=True,
	download=False,
	transform=transforms.ToTensor())
#print(train_data)

trainloader = torch.utils.data.DataLoader(train_data,
	batch_size=8,
	shuffle=True,
	num_workers=2)
test_data = torchvision.datasets.CIFAR10(root='./datasets',
	train=False,
	download=False,
	transform=transforms.ToTensor())
#print(test_data)
testloader = torch.utils.data.DataLoader(test_data,
	batch_size=8,
	shuffle=False,
	num_workers=2)

labels = ('plane','car','bird','cat','deer',
	'dog','frog','horse','ship','truck')

image_batch, labels_batch = iter(trainloader).next()
print(image_batch.shape)
img = torchvision.utils.make_grid(image_batch)
#plt.imshow(np.transpose(img, (1,2,0)))
#plt.axis('off')
#plt.show()
input_size = 3
hidden1 = 64
hidden2 = 128
output_size = len(labels)
k_conv_size = 5

class ConvNet(nn.Module):

	def __init__(self):
		super(ConvNet, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(input_size, hidden1, k_conv_size),
			nn.BatchNorm2d(hidden1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2))

		self.layer2 = nn.Sequential(
			nn.Conv2d(hidden1, hidden2, k_conv_size),
			nn.BatchNorm2d(hidden2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2))

		self.fc = nn.Linear(hidden2 * k_conv_size * k_conv_size, output_size)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)

		return out

model = ConvNet()
learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
total_step = len(trainloader)
epochs = 5

for epoch in range(epochs):
	for i, (images, labels) in enumerate(trainloader):
		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1)%2000==0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				.format(epoch+1, epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in testloader:
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print(correct/total * 100)
