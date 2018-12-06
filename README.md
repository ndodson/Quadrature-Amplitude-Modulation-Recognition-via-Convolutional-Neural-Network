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

The symbol represents the state of the waveform for a fixed period. In the 16qam example, the number of bits per symbol is 4, as the points in each quadrant can be grouped into fours. The advantage of higher bits per symbol is that there are more points within the constellation, making it possible to transmit more bits through the pipeline. This greatly increases the efficiency of transmission for communications systems. The downside of the higher order schemes is that the points in the constellation are closer together, making them more succestable to noise and cross-talk. 





<p align="center">
  <img width="460" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/64qam.7.png">
</p>




 


This is an example of a signal with low signal to noise ratio. As you can see, there is a lot of interference and it is difficult to tell the modulation scheme used. For this reason, the higher order methods are typically used when we know there is a high signal to noise ratio, resulting in less interference.


# Why do we care about constellation diagrams?

Constellation diagrams are important because they provide signal performance metrics of a communication system in a simple image. From the diagram, we can form an understanding of the modulation scheme, SINR, and signal preformance flaws. This data is conventionally analyized by an RF engineer, but in the development of intelligent systems and the software defined radio(SDR), we would like to automate this process for a real-time system. We will achieve this with a convolutional neural network.

# Goals

Through our implementation, we hope to accurately predict the modulation scheme used for low SINR constellations. 


# Dataset

Our data consists of 5,000 images of 4qam, 8qam, 16qam, 32qam, and 64qam consatellations. These constellations have SINR ranging from 10db to 30db. we will use one-hot-encoding to represent our data numerically. Then, we will resize and apply grayscale to our images. I chose to resize the images to 64*64. 

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

For this model, we have 3 convolutional layers and 3 pooling layers. We use a kernel size of 5 and stride of 1. The tensorboard visualization for our network is below.



<p align="center">
  <img width="460" height="460" src="https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/network_topology.png">
</p>





