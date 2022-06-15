from tkinter import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from train import train, Model
from classifier import classify_images
from display_results import visualize_train_loss, write_classigication_report_csv

class ImageDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.samples = data
        self.targets = targets
        self.transform = transform
    

    def __len__(self): return len(self.samples)


    def __getitem__(self, idx):
        image = self.samples[idx, :, :]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        
        return (
            image,
            torch.tensor(label, dtype=torch.int)
        )


def get_images_labels(file_name1, file_name2, file_name3):
    # the loading objects images
    object_1 = np.load(file_name1)
    object_2 = np.load(file_name2)
    sample = np.load(file_name3)
    
    # creating image labels
    object_1_labels = np.zeros(object_1.shape[0])
    object_2_labels = np.ones(object_2.shape[0])
    sample_labels = np.arange(sample.shape[0])
    
    # concatenating images/labels of the object1 and object2
    images = np.concatenate((object_1, object_2), axis=0)
    labels = np.concatenate((object_1_labels, object_2_labels), axis=0)
    
    # converting image ndarray to 3d image ndarray (similar to RGB image)
    # EX: (2000, 40, 60) -> (2000, 40, 60, 3)
    images = np.repeat(images[:, :, :, None], 3, axis=3)
    sample_image = np.repeat(sample[:, :, :, None], 3, axis=3)
    images = images.astype(np.uint8)
    sample_image = sample_image.astype(np.uint8)
    
    return (images, labels, sample_image, sample_labels)



if __name__ == "__main__":
    # Initializing parameters
    BATCH_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 200
    learning_rate = 1e-4
    target_names = ["object1", "object2"]

    # Getting and splitting datasets
    images, labels, sample_image, sample_labels = get_images_labels("object_1.npy", "object_2.npy", "sample.npy ")
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(data=images, targets=labels, transform=transforms)
    sample_dataset = ImageDataset(data=sample_image, targets=sample_labels, transform=transforms)
    train_set, val_set = random_split(dataset, [1600, 400])  # Splitting dataset to train and validation set (ratio 80/20%)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    sample_loader = DataLoader(dataset=sample_dataset, batch_size=200, shuffle=False)  # shuffle is False to not lose teh order ot images

    # Model initialization
    model = Model().to(device)
    # freezing all layers except the last fully connected layers
    # so that, model only learns the last dense layer
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # Model training
    torch.manual_seed(42)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_name = "model_for_classification.pth"
    hist = train(
        model, num_epochs,
        train_dl=train_loader, valid_dl=val_loader,
        loss_fn=loss_fn, optimizer=optimizer, 
        model_name=model_name, device=device
    )

    # Loading the trained model 
    # model = Model().to(device)
    # model.load_state_dict(torch.load('model_for_classification.pth'))
    
    # Classifing the images
    pred_val, y_val = classify_images(model=model, data_loader=val_loader, device=device)
    pred_train, y_train = classify_images(model=model, data_loader=train_loader, device=device)

    # Visualizing classifaction results
    # visualize_train_loss(hist, num_epochs)
    write_classigication_report_csv(y_true=y_train, y_pred=pred_train, target_names=target_names, file_name="train_report.csv")
    write_classigication_report_csv(y_true=y_val, y_pred=pred_val, target_names=target_names, file_name="validation_report.csv")

    # Classify the "sample.npy"
    predicted_test, _ = classify_images(model=model, data_loader=sample_loader, device=device)

    # Save prediction results of sample set
    np.save("predicted_results_for_sample.npy", predicted_test)
    print("sample prediction is written!")
