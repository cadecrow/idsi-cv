# Infrastrucutre Damage Analysis from Satellite Imagery using RCNN

## Abstract

In the aftermath of natural disasters, timely detection of where damages have happened and
dispatching rescue teams can save thousands of lives. To determine where efforts need to be focused, a human assessment of damage is usually required. These assessments take time and money and have inherent risk. To try and solve this problem, we worked on a two-step model that would take satellite images as input, detect buildings, and determine whether or not a building is critically damaged to warrant attention.

Utilizing a ResNet50 architecture, we were able to obtain high training accuracy on damage classification, above what we expect a human can achieve. However, the accuracy on our testing set showed that improvements can be made to reduce variance. Our attempt at using L2 weights regularization is promising and given the right parameter value it is expected that the model will not overfit to the training data and perform better on the testing data. For building detection, we use Mask R-CNN (based on FPN for object detection and RESNET101).
