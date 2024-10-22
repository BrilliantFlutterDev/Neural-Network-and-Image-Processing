# Neural-Network-and-Image-Processing

Introduction 

In the dynamic landscape of computer vision and image analysis, the convergence of techniques from neural networks and image processing has heralded a transformative era in image classification. The present study is wholly dedicated to an exhaustive exploration of this amalgamated strategy. This exploration encompasses the entire spectrum, from the formulation and execution of a classification model rooted in neural networks, tailored to a diverse image dataset, to the subsequent scrutiny of outcomes. Through the harmonisation of these robust methodologies, our intention is to harness their unified prowess, ultimately culminating in enhanced classification precision. During this research endeavour, we aspire to illuminate the intricate interplay that characterises neural networks and image processing. By doing so, we seek to contribute to a broader comprehension of their combined potential, specifically in the context of image classification tasks.

Image classification is the task of assigning a label to an image. This can be a challenging task, as images can be complex and contain a variety of features. Neural networks are a type of machine learning algorithm that can be used for image classification. Neural networks can learn to recognize patterns in images that are difficult for humans to see.

The objectives of this report:

Explore Synergistic Fusion: Investigate the harmonious amalgamation of techniques from neural networks and image processing, elucidating their joint efficacy in advancing the domain of image classification.

Formulate Effective Model: Devise and construct an image classification model rooted in neural networks, adeptly tailored to accommodate the inherent diversity within the chosen dataset of images.

Execute with Precision: Implement the devised model with optimal efficiency, ensuring resource utilization is judicious while achieving precise and expeditious image classification outcomes.

Assess Performance Metrics: Evaluate the model's classification accuracy and efficiency, engaging in a comprehensive comparison against established benchmarks, and unravelling the nuanced impact of image processing techniques.

Probe Complex Interaction: Analyse the intricate interplay between the realms of neural networks and image processing, meticulously examining how their synergistic collaboration enhances accuracy and facilitates deeper insights in image classification pursuits.

Contribute to Domain Knowledge: Furnish an all-encompassing scrutiny of the hybrid paradigm, furnishing valuable insights into the conceivable applications and far-reaching implications arising from the integration of neural network-based methodologies and image processing in the context of image classification challenges.

Overview
 This report presents the design, implementation, and evaluation of a neural network-based technique for image classification. The neural network achieved an accuracy of 100% on the Animals-10 dataset. This is a promising result, and it suggests that neural networks can be used to achieve high accuracy for image classification tasks.
In the first section of this course work, I will set objectives that what we must achieve after this its completion. In the second section I will set and explain the pre-processing which are needed before training and their results. In third section I will train the dataset and write explain the results I get and critical analysis.
 	
Simulation

Dataset Description 

	 The dataset I have selected for course work is Animal with the size of 614 MB and I downloaded it from kaggle. Then I have unzipped the folder and upload that folder into the MATLAB Drive. All the image sizes are not even so I have to resize these images before use for training. The total number of images that this dataset contains are 28k. 
I have selected mixture of images processing with artificial neural networks procedure. It contains 10 categories dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant. The images in the dataset have different backgrounds, same animal belong to a single category in different colours and breeds so when we train on this diverse image dataset the accuracy will improve. These images collected from “google images” and have medium quality.


As we can see the diversity in above screenshot the selection of ten specific animal categories demonstrates a strategic choice that encapsulates a mix of common and diverse animals. This selection might reflect considerations of animal diversity, recognition challenges, and potentially even ecological relevance. The dataset's variety enriches its utility across a range of applications. Images within the dataset exhibiting diverse backgrounds introduce an element of complexity. The presence of varied settings and environments requires the neural network to discern and prioritize animal features amidst potentially distracting elements. This variation challenges the model to capture robust features and ensures that it remains versatile enough to perform accurately in various contexts. This mirrors real-world scenarios where animals of the same species can exhibit considerable diversity. Addressing these variations within the dataset enhances the neural network's capacity to generalize effectively across diverse instances, improving its overall accuracy and reliability. Also, we can see that images procured from google images, a widely used source, possess varying degrees of quality. Curating a high-quality dataset might involve filtering out images that are blurry, overly pixelated, or irrelevant to the intended classification task. Ensuring image quality contributes to the model's ability to accurately identify key features.



Dataset Name 
Animals-10 
 
  Dataset Link 
 https://www.kaggle.com/datasets/alessiocorrado99/animals10 
Dataset Size 
 614 MB
Image 
Dimensions 
Image dimensions are different
Number of 
Classes 
 10 
Number of Images per 
Class 
cane 
cavallo 
elephant 
farfalla 
gallina 
gatto 
mucca 
pecora 
rango 
scoiattolo 
4863 
2623 
1446 
2112 
3098 
1668 
1866 
1820 
4821 
1862 
Notes (Any additional 
information you want to 
add) 
 Mixture of image processing with artificial neural networks (with MATLAB or Python) 



Encoding of the dataset

Download the dataset and upload the dataset on MATLAB Drive.
Get full path of the dataset that uploaded on MATLAB Drive by using following function:

	digitDatasetPath = fullfile( 'myfolder' , 'mysubfolder' , 'myfile.m' );

This MATALB function returns a character vector which containing the full path to the file which you want to load on MATLAB environment and use backslash ( \ ) as a file separator character .
Create an image datastore from the images in dataset by using following MATALB function:



This function creates an image datastore object in MATLAB environment. We can use this image datastore object to apply any processing technique on the dataset images. It loads all the images from a specific folder at once so that we don’t have to read images again and again from memory. Reading images every time you need can waste your system recourses. Steps 3 and 4 are used for reading the dataset.


Explore the dataset after reading it by using following MATLAB code:
figure;




randomperm function select random pictures from the dataset. subplot and imshow function to show images. The following screenshot is showed by the above code is showing that images in the dataset are of different sizes.







Pre-processing steps

There are several steps in this process from which we can apply all or few which depends upon our dataset. Some of them which I’m using in my code is as follows. 

Image resizing 

This function can resize the dataset images to a specific size so that training can be done smoothly.

Implementing traditional image processing techniques.

Filtering and convolution (e.g. blurring)
Edge Detection
Image Segmentation
Thresholding
Morphological Operations
Image Enhancement
Image Restoration
Feature Detection and Description 
Noise Removal 
Batch Processing
Background Subtraction
Erosion and Dilation
J)   Image Filtering

I used Filtering and convolution and Image Enhancement for image processing.

Filtering and convolution 
Gaussian filter smooth the images and reduce noise. It’s a convolutional filter. Gaussian kernel gives more weights to those pixels which are close to origin and less weights to those which are further away. It has some disadvantages as well that it can blur edges of images or can reduce sharpness.

In the following montage view left one is original image after applying Gaussian filter the noise in the image removed and smoothness increased.

 
Image Enhancement





In the above montage view the right one is after Image Enhancement. We can see that the right one is more visual. This function adjusts the brightness, saturation and contrast of images. It improves the appearance and make the images suitable for process.




We can use following feature extraction techniques.

Histogram of Orientation (HOG)
Scale Invariant Features Transform (SIFT)
Local Binary Patterns (LBP)
Colour Histogram
Gabor Filter
Principal Component Analysis (PCA)
Hough Transform

I used Histogram of Orientation (HOG) and Hough Transform in my code.

Histogram





Histogram of Orientation (HOG)

The Histogram of Oriented Gradients (HOG) is a frequently employed feature descriptor within the fields of computer vision and image manipulation. Its primary application lies in identifying objects in images, particularly in cases involving intricate backgrounds or diverse lighting circumstances. The underlying principle of HOG involves capturing the arrangement of gradient orientations at a local level within an image.
HOG demonstrates its effectiveness in situations where the detection of objects relies on vital shape and edge details, like in instances of identifying pedestrians or detecting faces.

Histogram of Oriented Gradients (HOG) aids me in pre-processing steps by emphasizing local gradient orientations, reducing noise, and enhancing relevant features in images. It addresses illumination changes, helps with object localization, and deals with pose variations. By simplifying data while preserving important information, HOG prepares images effectively for subsequent machine learning tasks like object detection and classification.








Hough Transform

The Hough Transform stands as a mathematical method utilized in the realm of image processing and computer vision to recognize shapes, specifically those that can be delineated through mathematical equations. It proves especially beneficial in pinpointing patterns such as lines, circles, and other geometric forms within an image, even when confronted with disturbances like noise and interruptions.

It helps me in the pre-processing steps by converting the task of shape detection in an image into a parameter space. Within this parameter space, every individual point corresponds to a conceivable shape present within the image.









Neural Network Architecture

Splitting the dataset:



Architecture:
The neural network architecture that I will use is a convolutional neural network (CNN). CNNs are a type of neural network that are specifically designed for image processing. CNNs extract features from images using a series of convolutional layers, and then classify the images using a series of fully connected layers.

The CNN architecture that I will use has the following layers:


The convolutional layers use the ReLU activation function, which allows the neural network to learn non-linear relationships between the features extracted from the images. The fully connected layers use the softmax activation function, which ensures that the output of the neural network is a probability distribution over the 10 classes.

Convolutional 2D layer: It conducts convolutions using adaptable filters on 2D input, such as images. This process involves sliding filters across the input, creating diverse feature maps. Through hierarchical learning, it identifies patterns ranging from edges to intricate features, playing a pivotal role in CNNs' proficiency for image recognition tasks.

Batch Normalization layer: The Batch Normalization layer is a crucial element in neural networks, essential for enhancing stability and training efficiency. It normalizes batch outputs, addressing internal covariate shifts and expediting convergence, ultimately reducing overfitting and bolstering model generalization. Its significance is particularly pronounced in deep networks, where it plays a pivotal role in optimizing performance.


ReLU: The Rectified Linear Activation (ReLU) is a pivotal activation function, introducing non-linearity in neural networks. It transforms negative input values to zero and retains positive values, enhancing gradient propagation during backpropagation. ReLU's simplicity and efficacy in preventing gradient vanishing have made it a prevalent selection for promoting efficient learning in deep networks while evading saturation.

MaxPooling: It is a critical process in convolutional neural networks, diminishes spatial dimensions while preserving crucial features. It achieves this by extracting the highest value from localized segments of the input, resulting in streamlined data and improved translation invariance. This technique effectively captures significant features, elevating the model's proficiency in detecting patterns of diverse scales.

Fully Connected layer: It constitutes a pivotal neural network element, establishing connections between each neuron and those in adjacent layers. Its function involves amalgamating extracted features, enabling nuanced pattern recognition and informed decision processes. This layer's comprehensive interconnectivity fosters the learning of intricate data relationships, enhancing the network's competence in tasks such as classification and regression.

Softmax layer: It is typically placed at a network's end for multiclass classification, converts raw scores into normalized probabilities. This transformation enables the model to assign class labels based on input data. By ensuring the probabilities sum to 1, Softmax identifies the most probable class for a given input.



Hyperparameters

The following hyperparameters were used for the neural network:


The learning rate controls how much the weights of the neural network are adjusted during training. A larger learning rate will cause the neural network to learn faster, but it may also cause the neural network to overfit the training data. A smaller learning rate will cause the neural network to learn slower, but it may be more likely to generalize well to unseen data.

The number of epochs is the number of times that the neural network is trained on the training data. A larger number of epochs will typically improve the accuracy of the neural network, but it will also take longer to train the neural network.

The batch size is the number of images that are fed into the neural network at a time. A larger batch size can improve the efficiency of training, but it can also make the neural network more difficult to train.

Results

In this report, we designed, implemented, and evaluated a neural network-based technique for image classification using MATLAB and the Animal-10 dataset. The neural network achieved an accuracy of 100% on the test data. This is a promising result, and it suggests that neural networks can be used to achieve high accuracy for image classification tasks.  The screenshot is given below.



Critical Analysis 

Here is a critical analysis of the above report, including some changes in the training code:

The neural network architecture is a good choice for image classification tasks. The convolutional layers can extract features from the images, and the fully connected layers are able to classify the images.
The hyperparameters are reasonable. The initial learning rate is small enough to avoid overfitting, and the maximum number of epochs is large enough to allow the neural network to learn the patterns in the data.
The training and evaluation results are promising. The neural network achieved an accuracy of 100% on the validation dataset. This is a good result, but there is still room for improvement.
The report could benefit from a more in-depth exploration of data augmentation techniques. Applying various transformations like rotations, flips, and brightness adjustments during augmentation can potentially enhance the model's ability to generalise to unseen data, warranting further investigation.
While the chosen neural network architecture showcases promise, the analysis could delve into whether utilising pre-trained models for transfer learning was considered. Integrating pre-trained models, particularly those trained on larger datasets, might expedite convergence and potentially lead to higher accuracy.
While the accuracy achieved is commendable, expanding the scope of evaluation metrics could offer a more comprehensive assessment. Incorporating metrics like precision, recall, and F1-score could provide deeper insights into the model's performance, especially considering potential class imbalances in the dataset.


 
Key takeaways:

Effective Architecture-Data Synergy: This report underscores the significance of aligning neural network architectures with the characteristics of the dataset. The choice of layers, nodes, and convolutional structures should resonate with the complexity and diversity of the data to achieve optimal feature extraction and classification.
Hyperparameter Fine-Tuning Impact: This report emphasizes the impact of hyperparameters on training outcomes. Through meticulous experimentation and tuning, it becomes evident that factors such as learning rate, epoch count, and batch size significantly influence the model's convergence and overall accuracy. 
Room for Continuous Improvement: Despite achieving promising results, this report highlights the notion that even successful models can benefit from further refinement. report underscore that there's always room for improvement, whether through augmenting data, incorporating regularization techniques, or experimenting with advanced architectures.
Holistic Evaluation Metrics: This report reinforces the importance of evaluating models through a holistic lens. While accuracy is a prominent metric, considering a range of evaluation metrics such as precision, recall, and F1-score provides a more comprehensive understanding of the model's performance, particularly in scenarios involving class imbalances.

Conclusion:

In this report, I designed, implemented, and evaluated a neural network-based technique for image classification using MATLAB and the Animal-10 dataset. The neural network achieved an accuracy of 100% on the validation dataset. This is a promising result, and it suggests that neural networks can be used to achieve high accuracy for image classification tasks.

There are several ways to improve the performance of the neural network. One way is to use a larger dataset. A larger dataset will provide the neural network with more information to learn from. Another way to improve the performance of the neural network is to use a more complex neural network architecture. A more complex neural network architecture will be able to learn more complex patterns in the images.

Overall, the results of this study are promising. Neural networks have the potential to be very effective for image classification tasks. Future work will focus on improving the performance of neural networks for image classification tasks.



Here are some changes that could be made to the training code:

The batch size could be increased. A larger batch size can improve the efficiency of training, but it can also make the neural network more difficult to train.
The learning rate could be decayed over time. This can help to prevent the neural network from overfitting the training data.
The dropout rate could be used. Dropout is a technique that randomly drops out some of the neurons in the neural network during training. This can help to prevent the neural network from overfitting the training data.
