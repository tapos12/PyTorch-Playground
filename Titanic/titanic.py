import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


dtype= torch.float
titanic = pd.read_csv('dataset/titanic/train.csv',
	sep = ',',
	engine='python',)
#print(titanic)
titanic_pred = pd.read_csv('dataset/titanic/test.csv',
	sep = ',',
	engine='python')

titanic = titanic.replace('', np.nan)
titanic_pred = titanic_pred.replace('', np.nan)
#print(titanic)


#col = ['Pclass','Sex','Age','SibSp','Parch','Embarked', 'Fare']
col = ['Survived','Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
titanic_features = titanic[col]
titanic_features = titanic_features.dropna()
titanic_target = titanic_features[['Survived']]

feature =['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
titanic_features = titanic_features[feature]
titanic_pred_features = titanic_pred[feature]

le = preprocessing.LabelEncoder()

titanic_features['Sex'] = le.fit_transform(titanic_features['Sex'])
titanic_pred_features['Sex'] = le.fit_transform(titanic_pred_features['Sex'])

titanic_features['Embarked'] = le.fit_transform(titanic_features['Embarked'])
titanic_pred_features['Embarked'] = le.fit_transform(titanic_pred_features['Embarked'])

titanic_features = pd.get_dummies(titanic_features, columns=['Pclass'])
titanic_pred_features = pd.get_dummies(titanic_pred_features, columns=['Pclass'])

#titanic_features[['Fare']] = preprocessing.scale(titanic_features[['Fare']])
#print(titanic_features[['Fare']].head())

x_train, x_test, y_train, y_test = train_test_split(titanic_features,
	titanic_target,
	test_size=0.2,
	random_state=0)

x_train_tensor = torch.from_numpy(x_train.values).float()
x_test_tensor = torch.from_numpy(x_test.values).float()
y_train_tensor = torch.from_numpy(y_train.values).view(1,-1)[0]
y_test_tensor = torch.from_numpy(y_test.values).view(1,-1)[0]

titanic_pred_features = titanic_pred_features.values
x_pred_tensor = torch.from_numpy(titanic_pred_features).float()

input = 7
output = 2
hidden = 50


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input, hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.fc3 = nn.Linear(hidden, output)

	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		x = self.fc3(x)

		return F.log_softmax(x, dim=-1)

model = Net()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.NLLLoss()

epoch_d = []
epochs = 3001

for epoch in range(1, epochs):
	optimizer.zero_grad()
	Ypred = model(x_train_tensor)

	loss = loss_fn(Ypred, y_train_tensor)
	loss.backward()

	optimizer.step()

	Ypred_test = model(x_test_tensor)
	loss_test = loss_fn(Ypred_test, y_test_tensor)

	_, pred = Ypred_test.data.max(1)

	accuracy = pred.eq(y_test_tensor.data).sum().item() / y_test.values.size
	epoch_d.append([epoch, loss.data.item(), loss_test.data.item(), accuracy])

	if epoch%100 ==0:
		print(epoch, loss.data.item(), loss_test.data.item(), accuracy)


torch.save(model, 'my_model_titanic')

final_pred = model(x_pred_tensor)
_, f_pred = final_pred.data.max(1)

f_pred_np = f_pred.data.numpy().reshape(-1,1)
df = pd.DataFrame.from_records(f_pred_np)
df.columns = ["Survived"]
#print(df)
merged = titanic_pred.join(df)
final = merged[["PassengerId","Survived"]]
final.to_csv('gender_submission.csv', sep=',', index=False)