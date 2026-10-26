import torch
import pandas as pd
import os

def predict_and_save(model, dataloader, file_name, task='binary', classes=['tmp']):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Ensure evaluation mode
    model.to(device)
    predictions = []
    labels = []

    with torch.no_grad():  # No gradient computation
        for batch in dataloader:
            inputs, batch_labels = batch[:2]  # Adjust based on your DataLoader
            if isinstance(inputs, list): inputs = [i.to(device) for i in inputs]
            else: inputs = inputs.to(device)
            batch_preds = model(inputs)
            if task == 'binary' or task == 'multilabel':
                batch_preds = torch.sigmoid(batch_preds) # binary/multilabel
            else:
                batch_preds = torch.softmax(batch_preds, dim=1)

            predictions.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    # df = pd.DataFrame({
    #     'Predictions': [p.item() for p in predictions],
    #     'Labels': [label.item() for label in labels],
    # })

    data = {}
    if len(classes) == 1:
        data[f'Prediction_{classes[0]}'] = [pred.item() for pred in predictions]
        data[f'Label_{classes[0]}'] = [label.item() for label in labels]
    else:
        for i, class_name in enumerate(classes):
            data[f'Prediction_{class_name}'] = [pred[i].item() for pred in predictions]
            data[f'Label_{class_name}'] = [label[i].item() for label in labels]

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, index=False)
