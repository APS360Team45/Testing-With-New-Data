import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import PIL
import math
from definitions import FruitRipenessDetector, evaluate

extensions = ['.jpg', '.jpeg', '.png', '.heic']

def process_imgs(folder): 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    imgs = []
    total_images = 0

    # Count the total number of images to process
    for fruit in os.listdir(folder):
        n_path = os.path.join(folder, fruit)

        if ".ini" in n_path:
            continue

        for ripeness, file in enumerate(os.listdir(n_path)):
            i_path = os.path.join(n_path, file)

            if ".ini" in i_path:
                continue

            img_list = os.listdir(i_path)

            for img in img_list:

                if ".ini" in img:
                    continue

                total_images += 1

    for fruit in os.listdir(folder):
        n_path = os.path.join(folder, fruit)

        if ".ini" in n_path:
            continue

        for ripeness, file in enumerate(os.listdir(n_path)):
            i_path = os.path.join(n_path, file)

            if ".ini" in i_path:
                continue

            label = ripeness
            img_list = os.listdir(i_path)

            for img in img_list:

                if ".ini" in img:
                    continue

                f_path = os.path.join(i_path, img)

                image = PIL.Image.open(f_path).convert("RGB")

                aspect_ratio = image.width / image.height

                if aspect_ratio < 1:
                    new_width = 256
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = 256
                    new_width = int(new_height * aspect_ratio)

                resized_image = image.resize((new_width, new_height))
                padded_image = PIL.ImageOps.pad(resized_image, (256, 256), color="black")

                hsv_image = padded_image.convert("HSV")

                imgs.append((transform(hsv_image), torch.tensor(label)))

    return imgs

def folder_to_pth(folder):
    data = process_imgs(folder)
    torch.save(data, f'dataset.pth')

def test_eval(net, loader, folder): 
    file_paths = []
    images = []
    total_images = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                image = np.asarray(Image.open(file_path))
                images.append(image)
                file_paths.append(file_path)
                total_images += 1
                
    # Create a figure and axis object for displaying the images
    num_images = total_images
    num_cols = min(num_images, 4)
    num_rows = math.ceil(num_images / num_cols)
    
    # Create a figure and axis object for displaying the predicted labels
    fig_pred, axs_pred = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3*math.ceil((total_images)/4)), squeeze=False)

    for i, (inputs, labels) in enumerate(loader):
        outputs = net(inputs)
        predictions = torch.round(outputs)
        
        for j in range(len(inputs)):
            
            prediction = int(predictions[j].item())
            actual = int(labels[j].item())

            image = inputs[j]
            prediction = predictions[j]
            
            print(f"Prediction: {prediction}, Actual: {actual}")

            axs_pred[j//4, j%4].imshow(images[j])
            axs_pred[j//4, j%4].set_title(f"Prediction: {int(prediction.item())}")
            axs_pred[j//4, j%4].axis('off')

    plt.savefig('predictions.png')
    
    fig_pred.savefig('predictions_pred.png')

MODEL_PATH = "model_ripeness_detector_bs64_lr0.001_epoch150"
TEST_PATH = "dataset.pth"
FOLDER = "photos"

folder_to_pth(FOLDER)

test_model = FruitRipenessDetector()
test_dataset = torch.load(TEST_PATH)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

paramys = torch.load(MODEL_PATH)
test_model.load_state_dict(paramys)

test_eval(test_model, test_loader, FOLDER)