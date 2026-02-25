# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

The objective is to predict future stock prices using historical closing price data.
The dataset consists of Google stock prices provided in:

trainset.csv

testset.csv

The model is trained using the training dataset and evaluated using the test dataset.
The closing prices are normalized using MinMaxScaler before training.

## Design Steps

### Step 1:
Import necessary libraries.
Load training and testing datasets.
Select closing price column.
Normalize data using MinMaxScaler.

### Step 2:
Convert time-series data into input-output sequences.
Define sequence length (60 days).
Convert data into PyTorch tensors.
Create DataLoader.

### Step 3:
Define RNN architecture.
Define loss function and optimizer.
Train the model.
Evaluate the model.
Plot actual vs predicted prices.


## Program
#### Name: GAYATHRI S
#### Register Number: 212224230073
```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the Model
epochs = 20
train_losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}')

print('Name: GAYATHRI S')
print('Register Number: 212224230073')
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.show()

model.eval()

with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Write your code here
print('Name: GAYATHRI S')
print('Register Number: 212224230073')
plt.figure(figsize=(10,6))

plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()

print("Predicted Price:", predicted_prices[-1][0])
print("Actual Price:", actual_prices[-1][0])
```

## Output

<img width="877" height="716" alt="Screenshot 2026-02-25 105208" src="https://github.com/user-attachments/assets/11426b6c-902a-4b1a-9c84-917d7282de1b" />

### True Stock Price, Predicted Stock Price vs time

<img width="1123" height="803" alt="Screenshot 2026-02-25 105234" src="https://github.com/user-attachments/assets/78f966b7-d719-42b3-8ade-d9c2d228a31f" />

### Predictions 
<img width="336" height="73" alt="image" src="https://github.com/user-attachments/assets/f2e9f4d9-0019-4929-aa00-26202a52da13" />

## Result
Thus, the Recurrent Neural Network (RNN) model for stock price prediction is successfully developed. The predicted stock prices closely follow the actual prices, demonstrating the effectiveness of RNN in modeling time-series data.

