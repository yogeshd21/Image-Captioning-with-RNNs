# Image-Captioning-with-RNNs
Trained different RNN (LSTM) architectures to implement image captioning model and study the respective outcomes on VisWis-Caption dataset.

Dataset Used: https://vizwiz.org/tasks-and-datasets/image-captioning/
The VizWiz-Captions dataset is a large-scale dataset, which includes: 23,431 training images with 117,155 training captions; 7,750 validation images with 
38,750 validation captions; 8,000 test images with 40,000 test captions (unavailable to developers; reserved for the image captioning competition). Here I am using a subset of the train images (8000 training images and 7750 Validation images) to train RNN/LSTM/GRU.

GLOVE word embedding model: https://www.kaggle.com/datasets/incorpes/glove6b200d

### Directory Format:

    .
    ├── glove.6B.200d.txt #Directory name
        ├── glove.6B.200d.txt #File name
    ├── model_weights
        ├── (All saved models will come here)
    ├── Pickle
        ├── encoded_test_images.pkl
        ├── encoded_train_images.pkl
    ├── VizWiz_Data_train1
        ├── train1
        ├── annotations_train1
    ├── VizWiz_Data_val
        ├── val
        ├── annotations_val
    ├── ImageCaptioning_Code.py
    └── Model_Ans.csv #Output CSV after Run iwth BLEU scores.

# Results

To study the performance with different models I tried 8 different models. Model 1 and 2 being the original model with1 LSTM layer having 0.5 and 0.2 dropout respectively, Model 3 and 4 is the model with 2 LSTM layers with 0.5 and 0.2 dropout respectively, Model 5 and 6 is the model with 2 LSTM layers and an added dense output layer with 0.5 and 0.2 dropout respectively, Model 7 and 8 are the models with 3 LSTM layers and added dense output layer with 0.5 and 0.2 dropout respectively. The performance for all the models was compiled into a datasheet which was as follows,

#### Model_Ans.csv

![image](https://user-images.githubusercontent.com/83297868/167320547-f006a44d-e8d5-4fbe-aeef-22b1d9007994.png)

The model with best BELU performance was the 7th Model with BELU 1 score as 0.480524 and BELU 2 score as 0.361183.

## Outcomes
![image](https://user-images.githubusercontent.com/83297868/167320647-9e45f75e-80d1-4449-bb97-8ded5e1b827d.png)
![image](https://user-images.githubusercontent.com/83297868/167320677-5f987576-e560-4726-af6c-314d2bf37353.png)
![image](https://user-images.githubusercontent.com/83297868/167320700-27d146dd-17fb-4e20-b3e4-b62a0ee50719.png)

