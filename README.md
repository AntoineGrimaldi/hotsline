# hotsline - Code for the paper **[A robust event-driven approach to always-on object recognition](https://laurentperrinet.github.io/publication/grimaldi-24/)**


Main contributions:

- Builds an adaptive, back to  back event-based pattern recognition architecture, inspired by neuroscience and capable of always-on decision, that is, that the decision can be taken it can be needed,


<img src="https://laurentperrinet.github.io/publication/grimaldi-24/hots.png" width="90%" />
The HOTS architecture.


- Extends the HOTS algorithm to increase its performance by adding a homeostatic gain control on the activity of neurons to improve the learning of spatio-temporal patterns, we prove an analogy with off-the-shelf LIF spiking neurons,

<img src="https://laurentperrinet.github.io/publication/grimaldi-24/DVSGesture_arm-roll.webp"  width="33%"/><img src="https://laurentperrinet.github.io/publication/grimaldi-24/DVSGesture_hand-clap.webp"  width="33%"/><img src="https://laurentperrinet.github.io/publication/grimaldi-24/DVSGesture_air-guitar.webp"  width="33%"/>


- Quantitative results assessing the role of homeostasis in the learning of spatio-temporal patterns and the performance of the network on different datasets: Poker-DVS, N-MNIST and DVS Gesture for different precisions of the temporal and spatial information.

<img src="https://laurentperrinet.github.io/publication/grimaldi-24/gesture_online.png" width="90%" />

Performance of the algorithm on the DVSgesture dataset. For this gesture recognition task, the online HOTS accuracy remains close to the chance level for about 100 events. More evidence needs to be accumulated, and then the accuracy increases monotonically, outperforming the previous method after about 10.000 events (at an average of 9.3% of the number of events in the sample).



![strangelove](https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:best/newscms/2016_51/1839926/161220-dr-strangelove-mn-1515.jpg)


