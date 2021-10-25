# Machine Learning for 3D Cloud Retrievals
Determining Cloud Optical Thickness from reflectance is a classical inverse problem. The rationale of the machine learning models for 3D cloud retrievals is given reflectance, we want to robustly predict Cloud Optical Thickness which produces the corresponding reflectance through machine learning models. In current physics-based method, researchers are using 1D retrieval to retrieve cloud properties based on the cloud’s 3D radiative transfer effects. But this method suffers from significant gap between retrieved cloud properties and real cloud properties. 3D radiative-transfer effects affect the radiance values and cloud properties for cloud retrievals.  Recent research has shown that deep learning models such as convolutional neural networks and deep feed-forward neural networks  can  reduce  retrieval  errors  for  multi-pixel cloud retrieval. However, these approaches are not robust.  In this study, we present various RNN-based deep learning models for cloud property retrieval. Bi-directional Long Short Term Memory with Transformer and Embedding achieves the best accuracy in predicting Cloud Optical Thickness through radiance values, with an distinct improvement from other deep learning algorithm.

In this study, we present various deep learning models with four algorithms, including Deep Feedforward Neural Network, Convolutional Neural Network, Long Short Term Memory,Bi-directional Long Short Term Memory, Bi-directional Long Short Term Memory with Transformer, and Bi-directional Long Short Term Memory with Transformer and Embedding. Bi-directional Long Short Term Memory with Transformer and Embedding achieves the best accuracy in retrieving Cloud Optical Thickness in both single-view and multi-view.

# BiLSTM with Transformer and Embedding model structure

![image](https://user-images.githubusercontent.com/55510330/137054721-aefb564e-3127-4f0f-a37d-8505dcb349f6.png)


## Contributors
* Xiangyang Meng, Department of Information Systems, University of Maryland Baltimore County, <xmeng1@umbc.edu>

* Sanjay Purushotham, Department of Information Systems, University of Maryland Baltimore County, <psanjay@umbc.edu>

## License
  Licensed under the [MIT LICENSE](LICENSE)
