import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import os
import zipfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,f1_score, accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt


class ECG_Data(Dataset):

    ''' 1 init call for the main zip file
        gititem returns 6-lead data, age, sex and the label of the ecg pattern
        we are using zipfile to first extract all the available filenames into a list which is parsed into extract_fildata to return the dataframe
    '''
    def __init__(self, zipfile_path, window_size):
        self.file_names = self.extract_filenames(zipfile_path)
        self.zipfile_path = zipfile_path
        self.window_size = window_size
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        file_name = self.file_names[index]
        data = self.extract_filedata(file_name)
        lead_data = data.iloc[:, 1:7].values.astype(float) #6 lead data , we can change to three leads too later on
        label = data.iloc[:, -1].values.astype(int) # 0s or 1
        age = data.iloc[:, -3].values.astype(int) #xxx digit number later to be normalized
        sex = data.iloc[:, -2].values.astype(int) # 0 for male and 1 for female 

        lead_scaling = MinMaxScaler()
        lead_data = lead_scaling.fit_transform(lead_data) #scaling the lead_data for optimized usage if deployed in a hardware setting
        
        age_scaling = MinMaxScaler(feature_range=(0,1))
        age = age_scaling.fit_transform(age.reshape(-1,1)).flatten() #normalizing age between 0 and 1

        windowed_lead_data = []
        for i in range(len(lead_data) - self.window_size + 1):              #windowing the lead data
            windowed_lead_data.append(lead_data[i:i+self.window_size])

            '''
                window size is basically the duration of the data we want, so since we have only 10 second raw data, we need to keep it 
                between 0 and 10, preferably between 5 and 1
                
            '''

        windowed_lead_data = torch.Tensor(windowed_lead_data)   #transforming to tensors
        label = torch.Tensor(label)
        age = torch.Tensor(age)
        sex = torch.Tensor(sex)

        return windowed_lead_data, label, age, sex

    def extract_filenames(self, zipfile_path):
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:        #used chatgpt for this bit involving zip since i have no experience with zipfile module
            file_names = zip_ref.namelist()                        #please check to both extract functions
        return file_names

    def extract_filedata(self, file_name):
        with zipfile.ZipFile(self.zipfile_path, 'r') as zip_ref:
            with zip_ref.open(file_name) as file:
                df = pd.read_csv(file)
        return df


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Model,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        #liner transformation converts lstm output to a liner one
        self.fc = nn.Linear(hidden_size, num_classes = 1) #i have two methods for this , a more better approach would be to set num_classes to 2
        self.sigmoid = nn.Sigmoid()  

        #right now num_class would return single value and threshold method can be applied (0.5)
        #but if we set it to 2 then we would have sigmoid values for both classes of arrythmia and normal heartbeat


    def forward(self, lead_data, age, sex):
        batch_size = lead_data.size(0)
        seq_length = lead_data.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)   #hidden state is the short term memory
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)   #cell state is the long term memory

        lstm_out, _ = self.lstm(lead_data, (h0, c0))

        last_output = lstm_out[:, -1, :] #taking the last hidden state activation for this

        age = age.unsqueeze(1).repeat(1, seq_length, 1)      #this is with chatgpt help, so please keep this on check
        sex = sex.unsqueeze(1).repeat(1, seq_length, 1)
        combined = torch.cat((last_output, age, sex), dim=2)
 
        output = self.fc(combined)   #applying final layers
        output = self.sigmoid(output)    

        return output


#model parameters

input_size = 8
num_layers = 2
epochs = 10
batch_size = 16
hidden_size = 128

dataset = ECG_Data(r'xyz.zip') 
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.BCELoss()
model = Model(input_size,hidden_size,num_layers)   
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
val_losses = []
train_losses = []
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    

    model.train()
    for lead_data, labels, age, sex in train_loader:

        optimizer.zero_grad()
        outputs = model(lead_data, age, sex)     #6+2 = 8 input features
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * lead_data.size(0)    #on the fence about this because im averaging out the lead losses of all of them instead of individually
    
    train_loss /= len(train_loader.dataset) 
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        for lead_data, labels, age, sex in val_loader:
            outputs = model(lead_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * lead_data.size(0)
        
    
    val_loss /= len(val_loader.dataset)  #average validation loss
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


# Testing Segment

model.eval()
test_loss = 0.0
prediction = []
ground_truth = []

with torch.no_grad():
    for lead_data, labels, age, sex in test_loader:
        outputs = model(lead_data, age, sex)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * lead_data.size(0)
        
        prediction.extend((outputs >= 0.5).int().tolist())      #sigmoid threshold
        ground_truth.extend(labels.tolist())

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")  



#VISUALIZING MODEL ACCURACY

accuracy = accuracy_score(ground_truth, prediction)
print('Accuracy: {:.2f}%'.format(accuracy * 100)) 
f1 = f1_score(ground_truth, prediction)
print(f"F1 Score: {f1:.4f}")                          
confusion_mat = confusion_matrix(ground_truth, prediction)
print(confusion_mat)

#plotting training segment losses 
plt.plot(train_losses, label='Training Loss', color = "r")
plt.plot(val_losses, label='Validation Loss', color = "g")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()