# Quadrature-Amplitude-Modulation-Convolutional-Neural-Network


Here we will implement a convolutional neural network to detect modulation schemes with low signal to noise ratio (SINR).
We will train the network on constellation diagram images ranging from low to high interference.


# First, what is Quadrature Amplitude Modulation(QAM)?



Amplitude modulation is a technique used in communications engineering to transmit data. It transports 2 digital bitstreams by changing the amplitude and phase of 2 sub-carrier waves. These 2 sub-carrier waves will be out of phase with each other by 90 degrees. The advantage of employing modulation in our communications systems that these modulated waves will be be lower bandwidth and lower frequency as compared to the carrier frequency. Essentially, this makes it possible to push more data through the pipeline with increased efficiency. 



There are many different versions of QAM that are widely used in digital communication. Some of the most widely used are 4qam(or qpsk), 16qam, and 64qam. These numerical prefixes represent the number of constellation points in a constellation diagram like below. 

![alt text](https://github.com/ndodson/Quadrature-Amplitude-Modulation-Convolutional-Neural-Network/blob/master/readme_images/16qam.75.png)


The symbol represents the state of the waveform for a fixed period. In the 16qam example, the number of bits per symbol is 4, as the points in each quadrant can be grouped into fours. The advantage of representing more sybols per bit is that there are more points within the constellation, making it possible to transmit more bits per symbol. This greatly increases the efficiency of transmission for communications systems. The downside of the higher order schemes is that the points in the constellation are closer together, making them more succestable to noise and cross-talk. For this reason, the higher order methods are typically used when we know there is a high signal to noise ratio.
