# SocialMedia_CNER
Code for ACL 2020 workshop paper "[Incorporating Uncertain Segmentation Information into Chinese NER for Social Media Text](https://arxiv.org/abs/2004.06384)".

Chinese word segmentation is necessary to provide word-level information for Chinese named entity recognition (NER) systems. However, segmentation error propagation is a challenge for Chinese NER while processing colloquial data like social media text. In this paper, we propose a model (UIcwsNN) that specializes in identifying entities from Chinese social media text, especially by leveraging uncertain information of word segmentation. Such ambiguous information contains all the potential segmentation states of a sentence that provides a channel for the model to infer deep word-level characteristics. We propose a trilogy (i.e., Candidate Position Embedding => Position Selective Attention => Adaptive Word Convolution) to encode uncertain word segmentation information and acquire appropriate word-level representation. Experimental results on the social media corpus show that our model alleviates the segmentation error cascading trouble effectively, and achieves a significant performance improvement of 2% over previous state-of-the-art methods.

### Reqirements:

* tensorflow>=1.8.0
* Keras>=2.2.0
* Python3
* Jieba
* keras-contrib

