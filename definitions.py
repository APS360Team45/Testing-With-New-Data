import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from plot import plot


class FruitRipenessDetector(nn.Module):
    def __init__(self):
         super(FruitRipenessDetector, self).__init__()
         self.name = "ripeness_detector"
         self.conv1 = nn.Conv2d(3, 50, 3, 1, 1) # in_channels = 3 (HSV), out_channels = 50 , kernel_size = 3x3, stride = 1, padding = 1 (to preserve resolution)
         self.pool = nn.MaxPool2d(2, 2) # max pooling for feature learning, repreated after every iteration
         self.conv2 = nn.Conv2d(50, 100, 3, 1, 1) # in_channels = 50 (output of conv1), out_channels = 100, everything else remains the same, keep adding +50 layers
         self.conv3 = nn.Conv2d(100, 150, 3, 1, 1)
         self.conv4 = nn.Conv2d(150, 200, 3, 1, 1) 
         self.conv5 = nn.Conv2d(200, 250, 3, 1, 1)
         self.conv6 = nn.Conv2d(250, 300, 3, 1, 1)
         self.fc1 = nn.Linear(300 * 8 * 8 , 32) # 300 * 8 * 8 output channels after conv6, reduced to 32 output features (arbitrary)
         self.fc2 = nn.Linear(32, 1) # 32 features reduced to 1 dimension for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 300 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x
    

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
e
    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path


def evaluate(net, loader, criterion, testing = False): # this function is for evaluating a model based on a given dataset and criterion
    '''
    net --> model
    loader --> type: DataLoader with specified batches
    criterion --> loss function
    '''
    total_loss = 0.0
    total_accuracy = 0.0
    total_iter = 0.0
    for i, data in enumerate(loader, 0):
      inputs, labels = data
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      total_loss += loss.item()

      if testing == True:
        corr_values = []
        for label, output in zip(labels, outputs):
            if label == 0:
                c_min, c_max = -0.5, 0.7
            elif label == 1:
                c_min, c_max = -1, 1
            elif label == 2:
                c_min, c_max = -0.8, 0.9
            elif label == 3:
                c_min, c_max = -0.9, 0.5

            c = output - label
            corr_values.append(1 if c_min < c < c_max else 0)

        corr = torch.tensor(corr_values).sum()
      else:
        prediction = torch.round(outputs) # round predictions to 0, 1, 2, or 3
        corr = torch.eq(prediction, labels).sum() # sum all the matching indeces together, obtaining a tensor containing boolean values, then summing them together

      total_accuracy += int(corr) # add number of correct predictions to total accuracy
      total_iter += len(labels) # update iteration by adding batch_size (number of labels)

    accuracy = float(total_accuracy)/total_iter # obtain accuracy by dividing total number of correct predictions by total number of predictions
    loss = float(total_loss) / (i + 1) # obtain loss by dividing total CE loss per batch by number of iterations
    return loss, accuracy


def train(model, train_dataset, val_dataset, batch_size=64, lr=0.001, num_epochs=10, print_stat=False, use_cuda=False, current_epoch=0):
    torch.manual_seed(1000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    train_loss = np.zeros(num_epochs)
    train_accuracy = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_accuracy = np.zeros(num_epochs)

    start_time = time.time()

    # training
    n = 0 + current_epoch # the number of iterations
    for epoch in range(num_epochs):
        n = n + 1
        running_loss = 0.0
        total_accuracy = 0.0
        total_iter = 0.0

        for i, data in enumerate(train_loader):
            imgs, labels = data
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            
            # convert the labels to floating point type
            labels = labels.float()

            out = model(imgs)
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            running_loss += loss.item()
            prediction = torch.round(out)# round predictions to 0, 1, 2, or 3
            corr = torch.eq(prediction, labels).sum() # sum all the matching indeces together, obtaining a tensor containing boolean values, then summing them together
            total_accuracy += int(corr)
            total_iter += len(labels)
        
        # save the current training information
        train_loss[epoch] = float(running_loss) / (i + 1)
        train_accuracy[epoch] = float(total_accuracy) / total_iter
        val_loss[epoch], val_accuracy[epoch] = evaluate(model, val_loader, criterion)

        print((f"Epoch {epoch + 1 + current_epoch}: Train accuracy: {train_accuracy[epoch] * 100:.1f}%, Train loss: {train_loss[epoch]:.4f} | "+
          f"Validation accuracy: {val_accuracy[epoch] * 100:.1f}%, Validation loss: {val_loss[epoch]:.4f}"))

        model_path = get_model_name(model.name, batch_size, lr, n)
        if (epoch+1) % 3 == 0:
            torch.save(model.state_dict(), model_path)

    end_time = time.time()
    torch.save(model.state_dict(), model_path)
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    # Save array of training/validation loss/accuracy
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

    np.savetxt("{}_train_accuracy.csv".format(model_path), train_accuracy)
    np.savetxt("{}_val_accuracy.csv".format(model_path), val_accuracy)

train_model = False # set to True if you want to train the model, False if you want to load a pre-trained model


version = 'v2/'


# if train_model:

#     test_model_0 = FruitRipenessDetector()
#     train_dataset = torch.load(f'{version}train_dataset_v2.pth')
#     val_dataset = torch.load(f'{version}val_dataset_v2.pth')

#     train(test_model_0, train_dataset, val_dataset, batch_size=64, print_stat=True, num_epochs=60)


#     # paramys = torch.load("model_ripeness_detector_bs64_lr0.001_epoch30")
#     # test_model_0.load_state_dict(paramys)

#     # test_loss, test_accuracy = evaluate(test_model_0, test_loader, nn.MSELoss())
#     # print(f"Test Accuracy (Epoch 30): {test_accuracy*100}%")


#     # train(test_model_0, train_dataset, val_dataset, batch_size=64, print_stat=True, num_epochs=30, current_epoch=30)

# test_model_extra = FruitRipenessDetector()

# if train_model == False:
#     paramys = torch.load(f"full-train-and-model\{version}model_ripeness_detector_bs64_lr0.001_epoch60")
#     test_model_extra.load_state_dict(paramys)


# test_dataset = torch.load('test_dataset_extra.pth')
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# test_loss, test_accuracy = evaluate(test_model_extra, test_loader, nn.MSELoss(), testing=True)
# print(f"Test Accuracy: {test_accuracy*100}%")
# print(test_loss)


# # Plot graphs

# plot_graphs = False

# if plot_graphs:
#     plot("model_ripeness_detector_bs64_lr0.001_epoch60")



########## Optional Fruit Specific Testing ##########

# fruit_specific = False

# if fruit_specific:
#     Banana_loader = torch.utils.data.DataLoader(torch.load('test_dataset_Banana.pth'), batch_size=64, shuffle=True)
#     Mango_loader = torch.utils.data.DataLoader(torch.load('test_dataset_Mango.pth'), batch_size=64, shuffle=True)
#     Tomato_loader = torch.utils.data.DataLoader(torch.load('test_dataset_Tomato.pth'), batch_size=64, shuffle=True)

#     Banana_loss, Banana_accuracy = evaluate(test_model_0, Banana_loader, nn.MSELoss())
#     Mango_loss, Mango_accuracy = evaluate(test_model_0, Mango_loader, nn.MSELoss())
#     Tomato_loss, Tomato_accuracy = evaluate(test_model_0, Tomato_loader, nn.MSELoss())

#     print(f"Banana Accuracy (Epoch 60): {Banana_accuracy*100}%")
#     print(f"Mango Accuracy (Epoch 60): {Mango_accuracy*100}%")
#     print(f"Tomato Accuracy (Epoch 60): {Tomato_accuracy*100}%")

