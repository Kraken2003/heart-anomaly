from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train(train_loader, test_loader, model, optimizer, loss_criteria=nn.BCELoss(), batchsize=batch_size, epochs=50):
    """
    Training loop for the model.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.
        loss_criteria (torch.nn.Module, optional): Loss function to compute training loss.
        batchsize (int): Batch size for training.
        epochs (int): Number of training epochs.
    """

    val_losses = []
    train_losses = []

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        min_loss = 1000.0

        model.train()
        for i, (lead_data, labels, age, sex) in enumerate(tqdm(train_loader)):
            lead_data = lead_data.to(device)
            age = age.to(device)
            sex = sex.to(device)

            # Convert labels to one-hot encoding
            labels = F.one_hot(labels, num_classes=2)
            labels = labels.type(torch.FloatTensor)
            labels = labels.squeeze()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(lead_data, age, sex)
            loss = loss_criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * lead_data.size(0)
            if loss.item() < min_loss:
                min_loss = loss.item()
            if (i + 1) % 50 == 0:
                print("EPOCH : {}/{}, MIN_LOSS : {}, LOSS : {}".format(epoch + 1, epochs, min_loss, loss.item()))

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        predictions = []
        actual = []
        with torch.no_grad():
            for lead_data, labels, age, sex in tqdm(test_loader):
                lead_data = lead_data.to(device)
                age = age.to(device)
                sex = sex.to(device)

                outputs = model(lead_data, age, sex)
                _, predicted = torch.max(outputs, 1)
                for i in range(labels.size(0)):
                    label = labels[i]
                    pred = predicted[i]
                    predictions.append(pred.item())
                    actual.append(label.item())
            cm = confusion_matrix(actual, predictions)
            print(classification_report(actual, predictions))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot()
            plt.show()
