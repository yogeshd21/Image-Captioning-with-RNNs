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
    └── Model_Ans.csv
