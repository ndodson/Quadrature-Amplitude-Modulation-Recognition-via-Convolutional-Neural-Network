### Table of Contents

- [Quadrature Amplitude Modulation](#quadrature)
- [Constellations](#constellations)
- [Interference](#interference)
- [Goals](#goals)
- [Dataset](#dataset)
- [Network Topology](#network)
- [Activations](#activations)
- [Results](#results)
- [Conclusion](#conclusion)
- [Credits](#credits)






# Quadrature-Amplitude-Modulation-Convolutional-Neural-Network


Here we will implement a convolutional neural network to detect modulation schemes with low signal to noise ratio (SINR).
We will train the network on constellation diagram images ranging from low to high interference.


# First, what is Quadrature Amplitude Modulation(QAM)?



Amplitude modulation is a technique used in communications engineering to transmit data. It transports 2 digital bitstreams by changing the amplitude and phase of 2 sub-carrier waves. These 2 sub-carrier waves will be out of phase with each other by 90 degrees. The advantage of employing modulation in our communications systems that these modulated waves will be be lower bandwidth and lower frequency as compared to the carrier frequency. Essentially, this makes it possible to push more data through the pipeline with increased efficiency. 



# Constellations

There are many different versions of QAM that are widely used in digital communication. Some of the most widely used are 4qam(or qpsk), 16qam, and 64qam. These numerical prefixes represent the number of constellation points in a constellation diagram like below. 



<p align="center">
  <img width="460" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/16qam.75.png">
</p>



# Interference

The symbol represents the state of the waveform for a fixed period. In the 16qam example, the number of bits per symbol is 4, as the points in each quadrant can be grouped into fours. The advantage of higher bits per symbol is that there are more points within the constellation, making it possible to transmit more bits through the pipeline. This greatly increases the efficiency of transmission for communications systems. The downside of the higher order schemes is that the points in the constellation are closer together, making them more susceptible to noise and cross-talk. 





<p align="center">
  <img width="460" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/64qam.7.png">
</p>




 


This is an example of a signal with low signal to noise ratio. As you can see, there is a lot of interference and it is difficult to tell the modulation scheme used. For this reason, the higher order methods are typically used when we know there is a high signal to noise ratio, resulting in less interference.


# Why do we care about constellation diagrams?

Constellation diagrams are important because they provide signal performance metrics of a communication system in a simple image. From the diagram, we can form an understanding of the modulation scheme, SINR, and signal preformance flaws. This data is conventionally analyized by an RF engineer, but in the development of intelligent systems and the software defined radio(SDR), we would like to automate this process for a real-time system. We will achieve this with a convolutional neural network.

# Goals

Through our implementation, we hope to accurately predict the modulation scheme used for low SINR constellations. Some of examples can be seen below. If possible, we would also like to train the model in as few epochs as possible and without a GPU, in order to simulate a real-time system without expensive hardware.

<p align="center">
  <img width="460" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/4qam.9.png">
</p>


# Dataset

Our data consists of 5,000 images of 4qam, 8qam, 16qam, 32qam, and 64qam consatellations. Due to lack of abundant image data, the size of the dataset was increased by augmenting images. The operations used were rotations, shears, horizontal/vertical flips, noise, and blur. These constellations have SINR ranging from -30db to 30db. we will use one-hot-encoding to represent our data numerically. Then, we will resize and apply grayscale to our images. I chose to resize the images to 64*64.  

```python

def one_hot_label(img):
    global ohl
    label = img.split('.')[0]
    if label == '4qam':
        ohl = np.array([0,0,0,0,1])
    elif label == '8qam':
        ohl = np.array([0,0,0,1,0])
    elif label == '16qam':
        ohl = np.array([0,0,1,0,0])
    elif label == '32qam':
        ohl = np.array([0,1,0,0,0])
    elif label == '64qam':
        ohl = np.array([1,0,0,0,0])
    return ohl
```

# Network Topology


We will follow a basic convolutional neural network architecture like below. We take an input image and apply a series of convolutional and maxpooling layers. We will apply an activation function, which will be discussed later, after each convolution. 

<p align="center">
  <img width="800" height="300" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/cnn.png">
</p>





For this model, we have 3 convolutional layers and 3 pooling layers. Convultion layer 1 applies 32 5 by 5 filters to the images. Layer 2 applies 50 5 by 5 filters to the image. Layer 3 applies 80 5 by 5 filters to the image. We use a kernel size of 5 and stride of 1. Dropout rate of 50% was added to avoid overfitting. The tensorboard visualization for our network is below. Output size of layer 1 is 64,64,32 with 832 trainable parameters. Output size of layer 2 is 13,13,50 with 40050 trainable parameters. Output size of layer 3 is 3,3,80 with 10080 trainable paremeters. The 2 fully connected layers after flattening will have output sizes of 1,512 with 41472 trainable parameters and 1,5 with 2565 trainable parameters. All together, there will be 184,999 trainable parameters.



<p align="center">
  <img width="800" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/network_topology.png">
</p>




# Activations

For our CNN, I decided to use tanh activation and softmax loss. 

Softmax loss is given by
<p align="center">
  <img width="400" height="100" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/softmax.png">
</p>

We apply the exonential fucntion on each coordinate, divided by the sum of all coordinates.

Tanh activation is given by

<p align="center">
  <img width="500" height="75" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/tanh.png">
</p>


We will apply Adaptive Moment Estimation(Adam) update as our means of gradient descent. Running averages of both the gradients and the second moments of the gradients are used. Weight and Loss are given by w and L respectively. The "forgetting factors" are given by B1 and B2.

<p align="center">
  <img width="700" height="350" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/Adam2.png">
</p>

Here is an example code showing the Adam update. v estimates of the 1st moment (the mean). cache estimates the 2nd raw moment (variance) 

<p align="center">
  <img width="700" height="150" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/Adam.png">
</p>


# Results


Below is a measure of the training accuracy after 4 epochs...

<p align="center">
  <img width="800" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/4epochs_acc.png">
</p>

Below is a mesure of the training loss after 4 epochs...

<p align="center">
  <img width="800" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/10epochs_loss.png">
</p>

Below, we can see that for 10 epochs, testing accuracy is 95%. This is due to overfitting from too many iterations. The red circle indicates the misclassified diagram.

<p align="center">
  <img width="800" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/overfitting_10_epochs.png">
</p>

Below is the result of 4 passes through the network and 100% testing accuracy. As you can see, diagrams with the red circle are the ones of most interest to us. The model is able to correctly classify 3 images that it has never seen before with very low SINR. We have now proven that a CNN can effectively classify the modulation schemes for very noisy conditions in a reasonable amount of time.

<p align="center">
  <img width="800" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/100%25test_acc.png">
</p>

 
# Conclusion

This model has proven to be applicable in a real time system. With only 2 passes through the network, 98% testing accuracy is achieved. In some cases, 100% is achieved, but the discrepancy is due to random weight intializations. If we set fixed weights, we could achieve 100% accuracy in 2 epochs, but our network will be biased. In order to consistently achieve 100% accuracy, 4 passes through the network are necessary. Other acitivation functions were implemented, such as ReLu and LReLu, but tanh gave the best results. Higher batchsize gave lower validation set accuracy. For a 3.5 gHz cpu, this takes roughly 1 minute and 30 seconds for a training set of 3,000 images. With a smaller dataset, faster training is possible, but there is no guarantee in accuracy. For 2 passes, the training time is about 45 seconds. In future iterations of this project, the model could easily be trained to not only predict the modulation scheme, but also give an estimation of the signal to noise ratio, given a prticular constellation diagram. 


# Credits

"Convolutional Neural Network Arhcitecture"
https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

"Activation Functions for Neural Networks"
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

"Quadrature Amplitdue Modulation"
https://www.osapublishing.org/DirectPDFAccess/682707A0-9527-0F82-AAE75127A49BF218_369137/oe-25-15-17150.pdf?da=1&id=369137&seq=0&mobile=no

Qinru Qiu CSE 400: Machine Intelligence with Deep Learning, Lecture 13. "Training Neural Networks"









