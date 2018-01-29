---
title: "Gold mine detection for mercury poisoning testing"
date: 29 January 2018
author: "Patrick Doupe, Matthew Le, Emilie Bruzelius, Bret Ericson, James Faghmous, Prabhjot Singh, Phil Landrigan"
output: pdf_document 
---

# Introduction

In this paper we discuss the methods used to identify potential gold mines in
the Western province of Ghana. We use a combination of free satellite images,
deep learning and unsupervised learning to acheive this. Our current results
underestimate the number of gold mines, aiming for high accuracy in the sites
selected. With more computing resources, we could scale this up to larger
regions.

# Method

To identify potential gold mines, we need to overcome two main challenges. 

1.  We do not have labels indicating the latitude and longitude of gold mines. 
2.  The areas covered are vast. 

For the first challenge, we identify gold mines 'by hand' and use supervised
learning. For the second challenge we focus on the Western province in Ghana and
use selection criteria that underestimates the detection of gold mines. 

The method consists of six steps.
1.  Download images from the region
2.  Convert these images to smaller images
3.  Extract information ('features') over these smaller regions
4.  Identify by hand a smaller image that shows a gold mine
5.  Identify images whose features are most similar to the image selected in 4.
6.  Repeat 5 with the images chosen in 5 as many times as required.

We now discuss these steps in detail.

## Download images from the region

Focusing on the Western Region in Ghana, we download satellite images from
[Bing maps](https://www.bing.com/maps).[^ToU] We do not know when the images
were taken nor which satellite. This is OK when we take this as a proof of
concept. First, the images are in the visible RGB spectrum, so we do not need to
know channel bandwidths. Second, can use the results in the mean time since we
can see that they 'work.' We will have to do some emailing for a paper version.

We downloaded 10,953 images. Each image is approximately 2000x1500 pixels in
size (width x height). For each image, we also store the image's polygon's
latitude and longitude in a GeoJSON file. This was originally stored in a
PostgreSQL database, but recently has been kept as a hard file.

## Convert these images to smaller patches

We convert these downloaded images into smaller patches for two reasons. First,
a large image has multiple characteristics. This can make it hard to extract
a single signal about what is happening on the ground an image is taken of. For
example, in a large image we may have some gold mines, forests and two towns.
Although another large image may have a gold mine, this larger image will not
look the same as the first image. It is easier with our method to focus on
small patches that show a town or a gold mine only. 

The second reason is that our feature extraction method is built on images with
a width by height by depth of 224 x 224 x 3. We therefore keep our patches to
have such a width by height. This yields 290,000 patches. We retain information
about the approximate latitude and longitude for each patch.

##  Extraction information ('features') over these smaller regions

The next step is to turn the 224x224x3 patches into single vectors that retain
the information in the image. 

We use a pre-trained convolutional neural network (CNN) to extract features.
CNNs are trained to identify objects in photos. They are used, amongst others,
to help social media platforms identify faces and objects in photos. We use a
particular CNN, called ResNet-50[^ResNet] to extract our features. ResNet-50
converts the 224x224x3 images into a 512 x 1 vector, reducing the information
300 times.

##  Identify by hand a patch that shows a gold mine

To identify a patch that shows a gold mine we first identify a gold mine on 
[Bing Maps](https://www.bing.com/maps). It is fortunate that gold mines are easy
to spot in satellite images. We take this latitude and longitude and using the
latitudes and longitudes from step two, identify the closest patch and call this
the _candidate patch_.

##  Identify patches most similar to the patch selected in step four.

We take the vector from step three and for all other patches' vectors, calculate
how similar each patch is to the candidate patch. As a similarity measure, we
use cosine similarity. 

$\mathbf {A} \cdot \mathbf {B}  \over \|\mathbf {A} \|\|\mathbf {B} \|$

Although cosine similarity is in common use in machine learning, alternative
metrics may perform better. We have not tested other metrics at this stage.

##  Repeat step five with the top patches chosen in step five as required.

In principal it is possible to calculate the a similarity measure for each patch
with each other patch. In practice it is takes too much time.[^sq] Instead, we
calculate the similarities for the top twenty patches from step five. We then have
at most twenty sets of twenty top patches (there is some overlap). We calculate
the top twenty patches for these sets. It is possible to iterate further of
course. 

# Results

The latitudes and longitude results are found in output.csv

[^ToU]: The ToU seem OK. See https://www.microsoft.com/en-us/maps/product/terms#anchor-1   
[^ResNet]: See https://arxiv.org/abs/1512.03385
[^sq]: We have (290000-1)^2 / 2 similarity measures to calculate, roughly fifty billion.




