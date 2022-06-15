import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def visualize_train_loss(hist, num_epochs):
    x = np.arange(1, num_epochs+1)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, hist[0], '-o', label='Train loss')
    ax.plot(x, hist[1], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x, hist[2], '-o', label='Train acc.')
    ax.plot(x, hist[3], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.savefig('Loss and Accuracy.png')
    plt.show()


def write_classigication_report_csv(y_true, y_pred, target_names, file_name):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(file_name, index=False)