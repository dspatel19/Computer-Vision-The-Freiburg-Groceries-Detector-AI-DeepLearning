<a name="br1"></a> 

COMPUTER VISION

1

Computer Vision: The Freiburg Groceries Dataset

Dhaval Patel

MSDS - 458 Artificial Intelligence & Deep Learning

Northwestern University



<a name="br2"></a> 

COMPUTER VISION

2

Abstract

The objective of this research is to examine the characteristics that make convolutional neural

networks effective in image classification, particularly in the retail sector, by assisting stores in

various industries in identifying objects, tracking customers, and enhance monitoring both

interior and outdoor situations. This would bring improvements and innovative technological

change in the shopping experience for both customers and retailers. By identifying each product

and the customer's activity, computer vision systems also aid in the prevention of stealing.

However, in order to apply CNN to this retail product recognition job, a substantial amount of

training data is required, which is very costly in practice. This is why we assessed and compared

several convolutional neural network models in 11 experiments to see how the researchers

conducted image classification. We used a multi-classification problem with 25 classes as the

output and a deep learning model trained with convolutional neural networks to categorize the

5000 images in the Freiburg Groceries dataset.



<a name="br3"></a> 

COMPUTER VISION

3

Introduction

It is revolutionizing how computer vision is used in practically every industry across the

world, including retail. More firms are investing in increasing their computer vision skills as they

grasp the potential of computer vision. From $9 billion in 2020, the worldwide computer vision

industry is expected to more than triple to $41 billion by 2030. Computer vision is being utilized

to increase omnichannel retail system efficiency (Javaid, 2022). In addition, Stanford researchers

developed a model to speed up grocery checkout processes. Most supermarkets apply plastic tags

with identification numbers to checkout items. This produces a large amount of plastic trash,

leaves potentially hazardous adhesive residue on delicate items such as fruits and vegetables,

needs a lot of physical work which can affect checkout chain efficiency (Ning et. al. 2018).

The Stanford researchers used the standard deep learning Convolutional Neural Network

(CNN) approach, which can only detect things at the visual level, but advanced semantic

segmentation techniques have limitations in differentiating different occurrences in the image.

YOLO, another object detection technology, detects things but not at the pixel level. These

concerns compel them to adopt a single Computer-Vision deep learning algorithm for example-

level classification and object recognition (Ning et al., 2018). Mask R-CNN is a category-leading

object-classification and instance-segmentation approach. The following diagram depicts the

high-level architecture of the mask R-CNN architecture:

(Fig.1)



<a name="br4"></a> 

COMPUTER VISION

Mask R-CNN essentially adds an extra branch to forecast an object mask. The additional mask

4

output differs from the class and box outputs that necessitates the extraction of a finer spatial 2

layout of an object. For example, segmentation difficulties are considered using Mask-RCNN as

it has shown to be beneficial in a variety of industries and domains.

Literature Review

This research was inspired by the popular Amazon Go superstore. The store operates

using an Amazon Go application utilizing computer vision, sensor fusion, and deep learning. You

enter Amazon Go, select the things you want, and then exit using the app. When things are taken

or returned to the shelves, this technology detects them and stores them in your virtual cart, and

your Amazon account is charged with a receipt emailed to you which is optimizing customer’s

shopping time experience with accuracy (Sagar, 2019).

Deep learning has grown in prominence during the last decade. Machine-learning

techniques, particularly convolutional neural networks are being used in the retail industry to

improve on check-in or out customer shopping services. CNN has been widely used in retail

product imaging classification, object identification, and semantic segmentation. Generally,

commercial robotic systems employ cutting-edge CNN algorithms to solve the challenge of

object recognition for various jobs. These approaches make use of existing deep learning models

by extracting 3D descriptors or by learning new feature representations from raw sensor data.

However, utilizing the full potential of machine learning technologies to address problems such

as grocery item recognition remains largely unrealized due to the lack of training data which is

one of the primary causes in this domain (Jund et al., 2016).



<a name="br5"></a> 

COMPUTER VISION

5

(Fig.2)

The researchers used their Freiburg grocery dataset to showcase their baseline architecture,

which comprises of five convolution layers and three completely fine-tuned connected layers.

The average accuracy reached by the researchers was 78.9%. It gives instances of successfully

identified images of various candy and pasta packaging, as seen in Fig. 2. Despite huge changes

in look, angles, lighting conditions, and the number of objects in each image, the neural network

can distinguish the categories in these images (Jund et al., 2016).

The breakthroughs in deep learning algorithms, as well as the open-source datasets available for

testing, are responsible for the CNN's image recognition performance. This project will

experiment with various techniques of training such a network using the Freiburg grocery dataset

to gain a better understanding of CNN in retail space.

Methods

Review Research

This research study explored a multi-classification problem with 25 output classes, where

the deep learning model learns to detect and classify images using CNN. CNN is comparable to

conventional neural networks in that it is composed of neurons with learnable weights, but it

varies from MLP as it relies on the weight sharing principle. There are three types of CNN

networks described in the literature: one-dimensional CNN (Conv1D), two-dimensional CNN

(Conv2D), and three-dimensional CNN (Conv3D). In addition, we used AlexNet neural network



<a name="br6"></a> 

COMPUTER VISION

model that suggested by the researchers as a baseline architecture to see if we can improve the

6

accuracy on predicting these retail product images.

Convolutional Neural Network Model Creation Process

CNN is driven by convolutional operation, which reduces the image while preserving its

features and removes noise. A feature map is generated by applying a convolution filter to

incoming data. The only parameters automatically learned during the training process are the

filters, while the size and number may be adjusted. Filters are 3x3, 5x5, or 7x7. Smaller filters

can extract more localized data from input. As we get further into the network, we may increase

the number of filters for a more robust model, but we run the danger of overfitting as the

parameter count rises.

Stride defines how far across the input the filter moves at each step. As the filter is moved

over the input, element-wise matrix multiplication is performed, and the results are then recorded

in the feature map. Zero values can be used as padding to reduce the size of feature maps by

surrounding the input with zeros.

Multiple convolution layers are conducted on an input, each applying a separate filter

and generating various feature maps. Following the convolution technique, we pass the output

via an activation function and pooling (usually, max pooling, which takes the maximum value in

the pooling window) to reduce the dimensionality of each feature map independently by

lowering the height and width while preserving the depth. This stage reduces the number of

parameters to save training time and prevent overfitting.

These intricate characteristics are learned by stacking convolution layers. Initial layers

detect edges or corners, subsequent levels combine this to recognize shapes, and future layers

integrate this information to conclude that this is an object's face. In a manner comparable to a



<a name="br7"></a> 

COMPUTER VISION

feed-forward network, we then flatten the output and add a couple of fully connected layers.

7

As shown in figure 3 below, we imported all the packages for data visualization, creation,

and testing for the CNN models.

(Fig.3 Packages)

Data Preparation, Exploration, Visualization Process

Using TensorFlow Keras, the Freiburg grocery dataset was loaded. The dataset contains 5,000,

images where each image size is 256 x 256 pixels, and comprising 25 distinct classes of

groceries, with at least 97 images per class shown in the bar plot and pie chart. The candy,

chocolate, juice, coffee, and tea are the top 5 categories with highest number of images in the

Frieburg grocery dataset out of the 25 distinct classes. Furthermore, we split 70% of the dataset

into 3115 labeled images for the training dataset and 30 % into 1485 labeled images for the

testing dataset. For validation, 347 images were selected from the training dataset. There is no



<a name="br8"></a> 

COMPUTER VISION

difference in label distribution between the three datasets. We will narrow the image size down

8

to 85 x 85 pixels during data ingestion and transformation process for faster model processing

time. Figure 4 depicts a random selection of 25 images categories and their labels.

(Fig.4 Sample Images)

The Figure 4 above contains examples of images from our dataset. Each images displays a single

or many instances of things with labels showing the 25 categories of food products.

Data Regularization and Compilation Process

The following techniques are used to standardize data: L2 regularization, L1

regularization, dropout rate, early stopping, and batch normalization are essential for preventing

model overfitting, model complexity, and vanishing or exploding gradients. In L2 (Ridge)

regularization, to minimize the loss function, the method must minimize both the original loss

function and the regularization term, which is proportional to the square of the weights. As the

value of lambda increases, the values of the parameters will drop since L2 penalizes them. In L1

(Lasso) regularization, an L1 penalty equal to the absolute value of the coefficient's magnitude is

added, or the coefficients' sizes are restricted. The difference between L1 and L2 regularize is

that the weights will not be sparse and L2 will obtain significantly more accurate results than

with L1. Similarly, the dropout rate eliminates certain nodes with all their incoming and outgoing

connections at each iteration by selecting them at random. Lastly, batch normalization approach



<a name="br9"></a> 

COMPUTER VISION

improves the performance and stability of neural networks by normalizing the inputs of each

9

layer. To prepare the network for training, it was necessary to develop the algorithms that would

be used to optimize the weights and biases depending on the data. We compiled the model with:

•

•

The sparse categorical crossentropy is commonly used and loss function provides the

difference between the predicted outputs and the actual outputs given in the dataset. The

goal is to minimize the difference between these two distributions.

The activation function used is Adam, which is a combination of RMSprop and

Stochastic Gradient Descent with momentum. It an optimizer that allows the network to

update itself depending on the data it sees and its loss function by keeping a moving

average of the squared gradient for each weight.

•

The softmax activation function of the last layer adds up the evidence that an input is in a

specific class by doing a weighted sum of the pixel intensity. The weight is negative if

that pixel having a high intensity is evidence against the image being in that class, and

positive if it is evidence in favor. Then, it converts that evidence into 25 probability

scores, summing up to 1.

Results

CNN and AlexNet Model Experiments Results (Fig.5)

We conducted 11 experiments as shown in the figure 5 above. The first two experiments we did

not use regularization for 2D CNN models with 3 convolutional and max pooling layers that



<a name="br10"></a> 

COMPUTER VISION

were running with batch size of 50 for variation in iteration from 20 to 50 epochs for all

10

experiments. For experiments 3 to 5, we used regularization such as L2 Regularizes, Dropout,

Early Stopping, and Batch Normalization. Furthermore, experimentation from 6 to 9 involved

tweaking hypermeters such as L1 Regularizes, Dropout rate, and increasing hidden to see if that

helps in improving the accuracy of the model in right direction. Finally, experiments 10 and 11

we built an AlexNet model based on the researchers baseline architecture with five convolution

layers and three completely fully connected layers to find best model with optimal accuracy,

faster processing time, and appropriate fitting of the data. But test accuracy of Alex Net model

was surprising, based results expectation compared to 2D CNN model with three convolution

layers. We will analysis the results of each experiment at a high level to understand how the

accuracy for the CNN models have improved over time based on variation in experiments.

In Experiment 1 and 2 we built 2D CNN model which is often used for image data with

3-dimensional input and output like the Freiburg grocery dataset. The CNN model consists of 3

convolutional layers 64, 128 and 128 nodes that has dense layers of 64 hidden neuron nodes in

the activation function, where with the following columns are utilized for training instead of all

the input feature dimensional space with no regularization, and we got accuracy of below 14%

for both experiments on test dataset it is underfitting in predicting those 25 classes. In addition,

there were only trainable around 749,465 parameters for experiment 1 and experiment 2 had

1,939,225 trainable parameters since we increase the hidden nodes of the layers which led to an

increase in the model complexity having more error amount in the surface, it should be simpler

and faster average processing time of 23 seconds to train the model in predicting the classes.

In Experiment 3, we built 2D CNN models with 3 convolutional layers using L2

regularizes to minimizes the loss function and penalize having around 749,337 trainable



<a name="br11"></a> 

COMPUTER VISION

parameters which are less as we increase lambda values and 30% dropout which removing nodes

11

every iteration to prevent overfitting the model led to worst CNN model accuracy of 9 %

accuracy on the test dataset compared to CNN models without regularization.

In Experiment 4, CNN model with 3 convolutional layers of 64, 128, and 128 hidden

nodes, and dense layers of 64 nodes but with L2 regularization, 20% dropout rate, and batch

normalization for image classification resulted that provided test data accuracy of 21.27% with

loss test error is 2.97 is which better improvement compared to first 3 experiments. In contrast,

train loss train error is 0.50 which shows that model overfitting with high variance, same

complexity of 749,337 with trainable parameters with built time of 23 seconds.

Best Final Model: Experiment 5 CNN3: (128, 256, 256)

and Dense Layer (512, 25) Plot (Fig. 6)

The Experiment 5, CNN model building process contains with 3 convolution layers 128,

256 and 256 nodes that has dense layers of 512 nodes following L2 regularize on dense layer

only, 20 % dropout, batch normalization, and early stopping regularization techniques were used

in the process of prevent model overfitting. The image classification resulted in test data of

41\.34% which is only slightly higher but much better optimization wise compared to other

experiments, but as we increase nodes in the convolution layers with L2 regularization which

helped with complexity in the model and appropriate fitting indicating but still high variance and



<a name="br12"></a> 

COMPUTER VISION

biases while analyzing the loss errors of dataset as well as based on the plot metrics from figure 6

12

shown above. The following experiment led to slower training time of 60 seconds because of 9.2

million trainable parameters compared to other models. The complexity of model is high and has

overfitting. The other experiments with L1 and AlexNet model were for investigating variations

towards improving accuracy, complexity, and performance of the CNN model.

Furthermore, we analyze the Experiment 5 by extracting the outputs from 2 filters of the

2 max pooling layers and visualize them in a grid as images as shown in figure 7 below.

(Fig.7)

We can analyze the ‘lighted’ up regions as shown above correspond to some features in

the original ‘cake’ images starting the convolution layer output followed by max pooling 2d

layers tries to extract stripe like features which were diagonal lines to intensely gather more local

information from input. The first layer seems to be responsible for detecting the edges of the cake

and background. The learning becomes more abstract as we go down the layers and they become

responsible for more specific detectors. Also, usually it’s easier to recognize the cake in the grid



<a name="br13"></a> 

COMPUTER VISION

images but 41% accuracy on the test dataset in predicting retails grocery product we are having

13

difficulty in visualizing the cake image in the grid.

We conducted further experimentation from 6 to 9 to see if we can get better optimal

accuracy with 2D CNN model following regularization to properly predicts all the 25 distinct

image classes. By tweaking hypermeters such as L1 Regularizes, variations in Dropout rate

layers from 20 to 30%, and doing variations of 3 convolutional layers 64, 128 and 128 nodes that

has dense layers of 64 nodes which decreased performance of the model by providing the test

accuracy between 7-35% on test data. The L1 regularization CNN models provided worst

accuracy of 7% compared to other models. Even though L1 regularization is the preferred choice

when having a high number of features as it provides sparse solutions but requires more training

data then we currently have in this dataset. Furthermore, test loss error is ranging high between

3\.01 – 4.46 which indicates low training error and high test loss error indicating that the model is

overfitting with high variance, biases, model complexity and built time with trainable parameters

is around as experiment 1 to 4.

Lastly, we decided to build a AlexNet CNN model in experiments 10 and 11 based on

the researcher’s demonstration on utilizing this dataset by following their baseline architecture of

five convolution layers and three fully connected layers. The expectation was getting better test

accuracy around 50-60% with AlexNet model which used total of 8 layers with optimized, and

regularized trainable parameters However, there were concerns of not fully understanding this

architecture from modeler side as we did get information about this architecture at a high level

from the researchers. But surprisingly we received a test of around 25-39% which is slightly

lower than our best model. But researchers getting optimal accuracy of around 78%, and our



<a name="br14"></a> 

COMPUTER VISION

models reaching test accuracy of around 41% with 3 convolutional layers is decent start to

14

conduct several more experiments in future on this dataset.

Finally, after doing several experiments we found an optimal best final model that we can

recommend to the business is Experiment 5 based on the results in figure 5. Below are the

confusion matrix evaluating the best and worst model based on different experiments.

Evaluate Model

Confusion Matrix (Best and Worst Model from Experimentation Results)

Fig.8 Best CNN L2 Model Confusion Matrix

Fig.9 Worst CNN L1 Model Confusion Matrix

The model was evaluated using the confusion matrix, which illustrates occurrences of a class

being incorrectly categorized. Comparing the confusion matrices for the best model and the

worst model (figures 8 and 9) reveals that a good model prediction would illuminate the values

on the diagonal line of the matrix, as seen in figure 8 for the CNN model with L2 regularization.

The second matrix demonstrates that the predicted labels do not correspond to the actual labels in

every instance; it is the confusion matrix for the CNN model with L1 regularization. The



<a name="br15"></a> 

COMPUTER VISION

accuracy of this CNN L1 regularize model on the test data and validation data was around 7%. In

15

the majority of instances, the training loss and validation loss curves indicated that the models

are underfitted. Therefore, this demonstrates that CNN models with L2 are well-suited for the 3-

dimensional input and output image data of the Freiburg grocery dataset. Lastly, we will display

the scatterplot shown in Figure 10 using t-Distributed Stochastic Neighbor Embedding (t-SNE),

an additional approach for dimensionality reduction that is particularly suitable for high-

dimensional datasets. Using this strategy, we decreased the number of hidden activation features

nodes in Experiment 5 using just the most significant input characteristics for training.

t-Distributed Stochastic Neighbor Embedding technique – Experiment 5 (Best Model)

(Fig.10)

The accuracy of our best recommended model from Experiment 5 is 41.34% which is depicted

in the scatter plot above using t-Distributed Stochastic Neighbor Embedding technique with 25

clusters shown above that are overlapping the models needs performance improvement. There is



<a name="br16"></a> 

COMPUTER VISION

a lot of overlap, most of them not clearly segmenting into classes with overlapping color code of

16

clusters this helps us in evaluating the model performance and accuracy. Precision and recall are

important evaluation metrics. The precision of the classifier confidence is the proportion of

recovered relevant instances, whereas the recall, sensitivity, or true positive rate is the true

positive rate of the prediction. Both numbers are in the range of 0 to 1, with 1 representing higher

prediction. The F-measure is a classification accuracy statistic that considers both accuracy and

recall values and presents the harmonic mean of these two values.

Conclusion

Knowing the benefits and limitations of CNN is crucial for real-world applications such

as retail products recognition imaging, with the goal of bring improvements and innovative

technological change in the shopping experience for both customers and retailers. CNN is

significantly easier to train than other deep learning algorithms and can detect critical traits with

high accuracy without the need for human intervention. The convolutional operation is the

CNN's powerhouse, responsible for building a simplified representation of the input image while

retaining all of its features and eliminating extraneous noise. CNN has the advantage of not

connecting the first convolutional layer to every pixel in the input image, which would be

computationally wasteful, but just to the pixels in the receptive fields. A Dense layer, on the

other hand, learns global patterns from all pixels in its input feature space.

Hence, I would recommend the management to use experiment 5 CNN model with 3

convolution layers 128, 256 and 256 nodes that has dense layers of 512 nodes following L2

regularize on dense layer only, 20 % dropout, batch normalization, and early stopping

regularization techniques were used in the process of prevent model overfitting as a base case

architecture to train using the Freiburg grocery data. Since it has an accuracy of 41.34% on test



<a name="br17"></a> 

COMPUTER VISION

dataset which leads to quicker training time of 60 seconds due to less 9.2 million trainable

17

parameters which is still a lot compared to other models and decent complexity level model that

fits well as a baseline architecture.

In the future, I hope to re-architect the following convolutional neural network that is

recommended above as well as understand the researchers baseline architecture of using

CaffeNet neural networks that consists of 5 convolutional layers, and 2 fully connected layers

with a more optimized approach of having less hidden nodes, which would reduce training time,

model complexity, and trainable parameters. By evaluating convolutional layer further by

tweaking hypermeters and experimenting with various regularization techniques to gather further

knowledge on having decent accuracy compared to our best optimal model is at 41% accuracy

that is currently recommended to the management. In addition, one of the primary causes of

having such a low accuracy is because of lack of training data for predicting between 25 distinct

grocery is really tough for the CNN models, and largely due to the grocery item recognition

remains unrealized as it requires a lot of improvements gathering such a image data.



<a name="br18"></a> 

COMPUTER VISION

18

References

Chollet, F. 2018. Deep Learning with Python. Shelter Island, N.Y.: Manning. [ISBN-13: 978-

1617294433]

Jund, P., Abdo, N., Eitel, A., & Burgard, W. (2016). The Freiburg Groceries Dataset. University

of Freiburg, Autonomous Intelligent Systems. Retrieved from http://ais.informatik.uni-

freiburg.de/publications/papers/jund16groceries.pdf.

Javaid, S. (2022). Top 5 Use Cases of Computer Vision in Retail. AIMultiple. Retrieved

December 2, 2022, from https://research.aimultiple.com/computer-vision-retail/

Ning, J., Li, Y., & Ramesh, A. (2018). Simplifying Grocery Checkout with Deep Learning.

Stanford University. http://cs230.stanford.edu/projects\_fall\_2019/reports/26257432.pdf

Sagar, A. (2019). Multi Class Object Classification for Retail Products. Towards Data Science.

Retrieved from https://towardsdatascience.com/multi-class-object-classification-for-

retail-products-aa4ecaaaa096.



<a name="br19"></a> 

COMPUTER VISION

19

Appendix

Showcasing only Top 3 Performing Models out of 11 Experiments

Experiment 5 (Final Best Model): CNN 3 Layer with (L2 Reg, Dropout 20%, Batch Norm)

(Hidden Nodes: (128,256, 256), Dense Layer: (512,25))



<a name="br20"></a> 

COMPUTER VISION

20

Experiment 9: CNN 3 Layer with (L1 Regularize, Dropout 20%, Batch Norm)

(Hidden Nodes: (128,256, 256), Dense Layer: (512,25))



<a name="br21"></a> 

COMPUTER VISION

21

Experiment 11: AlexNet Model CNN 5 Layer with 3 Fully Connected Layers (L2 Reg, Dropout

20%, Flatten, and Batch Norm)

