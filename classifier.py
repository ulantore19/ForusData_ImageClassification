import torch


def classify_images(model, device, data_loader=None):
    predicted_values = torch.tensor([]).to(device)
    true_values = torch.tensor([]).to(device)

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            predicted = torch.round(torch.sigmoid(y_hat.squeeze()))
            y = torch.round(torch.sigmoid(y.squeeze()))
            predicted_values = torch.concat([predicted_values, predicted], dim=0)
            true_values = torch.concat([true_values, y], dim=0)
    
    return (predicted_values.cpu().numpy(), true_values.cpu().numpy()) 