import torch
import torch.nn as nn
import pandas as pd

# criando a base dos dados
temp_celcius = [-10.0, 20.0, 100.0]
temp_farenheit = [14.0, 68.0, 212.0]

# colocando em uma tabela
df = pd.DataFrame({'Celcius': temp_celcius, 'Farenheit': temp_farenheit})
print(df)

# modificando as variáveis
x = torch.FloatTensor(df.Celcius.values.astype(float))
y = torch.FloatTensor(df.Farenheit.values.astype(float))
y = y.unsqueeze(1)

'''Modelo de ML'''
class Model(nn.Module):

    '''um neuronio de input e um de output, usando o bias'''
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        out = self.input_layer(x)
        return out
    
EPOCHS = 1000
LR = 0.2

model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

x = x.view(x.size(0), -1)

pesos = []
bias = []
# treinamento
for epoch in range(EPOCHS):

    outputs = model.forward(x)

    loss = criterion(outputs, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    pesos.append(model.input_layer.weight.item())
    bias.append(model.input_layer.bias.item())

print(f'Peso: {model.input_layer.weight.item():.2f} Bias: {model.input_layer.bias.item():.2f}')

#exportando cada peso e bias por época
training_data = pd.DataFrame({'Pesos':pesos, 'Bias':bias})
training_data.to_csv('training_data.to_csv', index=False)
print(training_data)

valor_para_calcular = float(input('Digite um número em celcius: '))
print(f'Valor predito: {model.forward(torch.FloatTensor([valor_para_calcular])).item()}')
print(f'Valor real: {1.8*valor_para_calcular + 32}')
