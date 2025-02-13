import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(
    train_dataset, 
    val_dataset,
    title:str,
    plot_heatmap:bool = True,
    batch_size: int = 20, 
    num_of_epochs: int = 10,
    model=None,  # nn architecture
    criterion=None, # Loss function
    optimizer=None,  # How the step function will update the weights
):
    """
    Trains the model, evaluates it on the validation set at every epoch,
    and plots learning curves and a confusion matrix at the end.
    
    Parameters:
      - train_dataset: a TensorDataset for training
      - val_dataset: a TensorDataset for validation
      - batch_size: mini-batch size for DataLoaders
      - model: your model instance (e.g., an MLP)
      - criterion: loss function (e.g., CrossEntropyLoss)
      - optimizer: optimizer (e.g., Adam with model.parameters())
      - num_of_epochs: number of training epochs
    
    Returns:
      A dictionary containing the training/validation losses and accuracies, and the confusion matrix.
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_of_epochs):

        # training 
        model.train()  
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad() 
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            # loss and accuracy 
            running_train_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
        
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # validation
        model.eval()  
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                outputs = model(val_data)
                loss = criterion(outputs, val_labels)
                running_val_loss += loss.item() * val_data.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # Record the metrics
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
    # Plotting 
    epochs = np.arange(1, num_of_epochs + 1)
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    plt.suptitle(title, fontsize=25)

    # Plot Losses
    ax[0].plot(epochs, train_losses, label='Train Loss')
    ax[0].plot(epochs, val_losses, label='Validation Loss')
    ax[0].set_xlabel('Epoch', fontsize=15)
    ax[0].set_ylabel('Loss', fontsize=15)
    ax[0].set_title('Loss Learning Curve', fontsize=20)
    ax[0].legend()
    
    # Plot Accuracies
    ax[1].plot(epochs, train_accuracies, label='Train Accuracy')
    ax[1].plot(epochs, val_accuracies, label='Validation Accuracy')
    ax[1].set_xlabel('Epoch', fontsize=15)
    ax[1].set_ylabel('Accuracy', fontsize=15)
    ax[1].set_title('Accuracy Learning Curve', fontsize=20)
    ax[1].legend()
    
    # confusion matrix
    all_preds = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            outputs = model(val_data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    if plot_heatmap:
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(title, fontsize=15)
        plt.show()
    

