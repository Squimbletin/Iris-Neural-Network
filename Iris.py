import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define Model
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x

# Seed for reproducibility
torch.manual_seed(32)

# Load dataset
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# Strip column names to avoid hidden spaces
my_df.rename(columns=lambda x: x.strip(), inplace=True)

# Normalize the 'species' names (ensure consistent casing)
my_df['species'] = my_df['species'].str.strip().str.title()

# Ensure the mapping works
mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
my_df = my_df[my_df['species'].isin(mapping.keys())]  # Keep valid rows

# Proceed with mapping the species to numerical values
my_df['species'] = my_df['species'].map(mapping)

# Drop NaN values if any
my_df = my_df.dropna()

# Prepare training data
x = my_df.drop('species', axis=1).values
y = my_df['species'].values.astype(int)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # Ensure int64
y_test = torch.tensor(y_test, dtype=torch.long)

# Model and Training Setup
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 100
losses = []
for i in range(epochs):
    # Forward pass
    y_pred = model.forward(x_train)

    # Compute loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i}, Loss: {loss.item()}')

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss over epochs
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)

# Display the plot
plt.show()

#Evaluate Model
with torch.no_grad(): # turn off back Prop
    y_eval = model.forward(x_test) #Predictions based of 20% test data
    loss = criterion(y_eval, y_test) #Find the loss
    print(f'Test Loss: {loss.item()}')

    Correct = 0
    with torch.no_grad():
        for i, data in enumerate(x_test):
            y_val = model.forward(data)
            #what network thinks it is
            print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')
            if y_val.argmax() == y_test[i]:
                Correct += 1

    print(f'Test Accuracy: {Correct, Correct / len(x_test)*100}')

