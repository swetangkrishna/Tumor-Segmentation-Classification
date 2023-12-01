Classification Model:
In this code is a use case of deep learning in the medical field. The task here is to classify the ultrasound scans into 'benign', 'melignant' and 'normal'. The second use case is to identify tumors in ultrasound scans.
 Our Data set comprises ultrasound scans stored as Numb by arrays.  Each scan is an RGB image of size 128 X128 the data set is divided into three categories, benign, melignant and normal. 'Input.npy' file contains ultrasound scansand  'target.npy' file contains their respective labels.

 After loading the data we structure the data by loading the input scans and target maps. Split the data into training and test sets given limited size of data, data argumentation is a key technique to artificially expand the training set. Various transformation techniques are used in this model such as rotation, contrast enhancing, blurring, flipping and shear transformation.

For this model we use transfer learning technique and import 'MobileNetV2'. It is built on the concept of inverted residuals and linear bottlenecks. After adapting 'MobileNetV2' we add custom layers on top of it . This includes global average pooling layer followed by a dense layer with 1024 units, a Dropout layer for regularization and finally soft Max with three categories.

While training the model I used early stopping to avoid over fitting. I utilized a dynamic learning rate that decreases when the validation loss plateaus allowing the model to make finer adjustments to the weights and converge to better performance.

Optimizer used is 'Adam' known for its efficiency and adaptability. The loss function for the classification problems used here is categorical class entropy where each class is mutually exclusive.

The graph model 'loss' depicts a downward trend for both training and validation loss, the smaller gap between the training and validation loss indicates that the model is performing well for unseen data. The accuracy graph reflects an upward trajectory for both training and validation sets.

Segmentation Model:
 Our model is based on U-Net architecture which is specifically designed for biomedical image segmentation. To adapt unit for our data set we implement on encoding network that captures the features at different resolution, followed by a decoding network that reconstructs the segmentation. This U-Net is designed with a series of convolutional and pooling layers that constitute the down sampling path.

Fetching the data set, firstly we read and separate numpy files for each category of ultrasound scans and their corresponding segmentation maps. 

The 'Build_model' function constructs the unit model used for segmentation medical images.  This model consist of encoding blocks that down sample the input images while extracting features. A bridge layer processes the most reduced feature map decoding blocks, then the decoding block unsamples these feature maps to the original image size, using skip connections from encoding blocks to preserve spatial information.


Think of the encoding blocks as a series of steps that both simplify and analyze the image. Simplifying the image, the image is made smaller, imagine zooming out of a photo. Analyzing the image, while the images being made smaller the network is also learning what's important in the image after passing through several encoding blocks and has become much smaller it passes through a bridge layer that layer further processes the image focusing on the most important feature identified so far. The Decoding block do the opposite of the encoding blocks they gradually make image again bigger zooming back into the image and adding back the details lost while zooming out.

As the image is made bigger it uses information from the encoding blocks to help restore the image.

The last part of the network takes the processed image and decides for each pixel if it's part of tumor or not. 

Compiling the model and preparing the data for training. Here the model uses the 'Adam' optimizer a popular choice for its efficiency and adaptability in training neural nets loss function is binary cross entropy suitable for binary classification there are also custom metrics created such as F1 score recall and precision. 

Data augmentation enhances the data set by creating modified version of the images.  The transformation like rotation shifting and flipping are applied to the images and their masks.

 Early stopping based on F1 score this callback monitors the models F1 score on the validation set. For the result validation, we use recall; that is correctly identifying all instances of tumors, precision; areas identifies as tumors are actually tumors. Then, calculates F1 score which is the balanced measure to  judge the performance of our model.

Challenges that were faced in creation of this model were overfitting, class imbalance, small data set and overtraining. 

These challenges were overcome by using various techniques such as data argumentation, early stopping, batch normalization and regularization. 