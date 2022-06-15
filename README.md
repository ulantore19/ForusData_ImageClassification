# Instructions

There are 3 files with the extension `.npy` in this folder.
They can be read with `numpy`.

- `object_1.npy`: Sample images from class A
- `object_2.npy`: Sample images from class B
- `sample.npy`: Sample images that are similar to the images that will be used for evaluation. Class labels are not provided.

Please write in python:

1. a training program that generates a model to be used by the classifier program,
2. a classifier program that uses the model above to classify images as object_1 or object_2, and
3. an evaluation program that evaluates the performance of the classifier program above.

Please, pay close attention that your program use Pytorch, and has no more than 10M parameters.


# Method

In order to classify the given images the transfer learning was applied. Particularly, Efficientnet-b2 was chosen as the base model
with 9.2M paramteres. You can install the pretrained model from source or via pip:
```
pip install efficientnet_pytorch
``` 
or:
```
git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e .
```

The predicted results of the images in `sample.npy` are stored in `predicted_results_for_sample.npy`.


## Results
Trained model is saved in "model_for_classification.pth"
The top validation accuracy is **93.5%**. 
The plot of how accuracy and loss values have been chaning displayed in figure below.
![Loss and Accuracy values on train and validation set](Loss%20and%20Accuracy.png)
Prediction of "sample.npy" is written in "predicted_results_for_sample.npy". The metric values were written in `validation.csv`.
