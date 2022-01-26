# Machine Learning for 3D Cloud Retrievals
In this GitHub repositary, you will find information on how to preprocess the raw data, how to install required packages, all the deep learning models,how to train and test the models, and how to evaluate the test results quantitively and through visualization.

## Abstract
Retrieving cloud microphysics or optical properties  from reflectance is an inverse problem.  Traditional physics-based method uses 1D retrieval to retrieve cloud properties based on the cloud’s three-dimensional (3D) radiative transfer effects. But this method suffers from significant gap between retrieved cloud properties and real cloud properties since 3D radiative-transfer effects affect the radiance values and cloud properties for cloud retrievals.  Recent researches have shown the feasibility of using deep learning models, such as convolutional neural networks and deep feed-forward neural networks, to reduce  retrieval  errors  for  multi-pixel cloud retrieval. In this study, we present various RNN-based deep learning models for cloud properties retrieval. Bi-directional Long Short Term Memory with Transformer and Embedding achieves the best outcome in retrieving Cloud Optical Thickness and Cloud Effective Radius through reflectance values, with a distinct improvement from other deep learning algorithms.


## Datasets
Three versions of datasets summerized in Table 1:

* Version 1: simple fractal cloud for COT only with constant cloud top 

11 Features: Solar Azimuth Angle, Solar Zenith Angle, Surface Albedo, View Zenith Angle, Liquid water path, Effective radius, Wavelength, reflectance values, Cloud Optical Thickness, spatial x, spatial y

* Version 2: simple fractal cloud for COT only with varying cloud top

13 Features: Solar Azimuth Angle, Solar Zenith Angle, Surface Albedo, View Zenith Angle, Liquid water path, Effective radius, Wavelength, reflectance values, Cloud Optical Thickness, spatial x, spatial y, cloud_top_height(km), cloud_top_height(meters).

* Version 3: COT and CER with varying cloud top and two wavelengths

11 Features: Solar Azimuth Angle, Solar Zenith Angle, Surface Albedo, View Zenith Angle, Liquid water path, Effective radius, Wavelength, reflectance values, Cloud Optical Thickness, spatial x, spatial y


![1640818610(1)](https://user-images.githubusercontent.com/55510330/147709506-c3cb1ecd-f3d1-45e5-babb-9aa32bc09041.png)

We use Dataset 1 and Dataset 2 for Cloud Optical Thickness retrieval: 

![WeChat Image_20211025153436](https://user-images.githubusercontent.com/55510330/138758886-be31f8ea-d4fd-42da-ac03-eb717bc92703.png)
## Pipeline
![pipeline](https://user-images.githubusercontent.com/55510330/149815510-5dbae0b2-6530-47c4-b597-6e27546f22d4.png)
### Dataset Preparation
As shown in dataset description above, each dataset is consisted of synthetic cloud profiles, and each synthetic cloud profile contains radiance values and COT values. For example, each demo cloud profile here has 4096 radiance values (input for deep learning models) and 4096 COT values （target output for deep learning models). The current resolution is 10m. We average reflectance data to 500m resolution and 1000m resolution before feeding them to the deep learning models. As a result, there are 82 (4096/50≈82)radiance values and 41(4096/100≈41) radiance values in each synthetic cloud profile, respectfully. The dataset preprocessing code prepares the dataset which will be used for the following deep learning models training and testing. For dataset preprocessing code, please refer to `3D_cloud_retrieval/COT retrieval_Dataset1&2/Dataset_preparation.py`. 

### Models Training

The deep learning models for retrieving Cloud Optical Thickness are as follows:

A. DNN-based model (Okamura et al.) `3D_cloud_retrieval/COT_retrieval_Dataset1&2/DNN.py`

B. CNN-based model `3D_cloud_retrieval/COT_retrieval_Dataset1&2/CNN.py`

C. Our proposed RNN-based models:
* BiLSTM with Transformer and Embedding `3D_cloud_retrieval/COT_retrieval_Dataset1&2/BiLSTM_Transformer_embedding.py`
### Structure of deep learning model of retreiving COT: BiLSTM with Transformer and Embedding
![BiLSTM transformer embedding (1)](https://user-images.githubusercontent.com/55510330/151221487-98c05139-8bb8-4af9-9158-155f415e9f00.png)
* BiLSTM with Transformer `3D_cloud_retrieval/COT_retrieval_Dataset1&2/BiLSTM_with_Transformer.py`
* Transformer `3D_cloud_retrieval/COT_retrieval_Dataset1&2/Transformer.py`
* LSTM with Transformer `3D_cloud_retrieval/COT_retrieval_Dataset1&2/LSTM_with_Transformer.py`
* BiLSTM with Embedding `3D_cloud_retrieval/COT_retrieval_Dataset1&2/BiLSTM_embedding.py`
* LSTM with Embedding `3D_cloud_retrieval/COT_retrieval_Dataset1&2/LSTM_embedding.py`
* BiLSTM `3D_cloud_retrieval/COT_retrieval_Dataset1&2/BiLSTM.py`
* LSTM `3D_cloud_retrieval/COT_retrieval_Dataset1&2/LSTM.py`

Bi-directional Long Short Term Memory with Transformer and Embedding achieves the best outcome in terms of COT retrieval. Hyperparameters fine-tuning is not included in these models as this process was done off-line. To train any of the above RNN-based model, you may run `3D_cloud_retrieval/COT retrieval_Dataset1&2/Main.py` The trained model will be saved and used for testing by running `3D_cloud_retrieval/COT retrieval_Dataset1&2/test.py`



The deep learning model to retrieve COT and CER together is as the following:
* `3D_cloud_retrieval/CER and COT retrievals_Dataset3/CER_and_COT_retrievals_DL model.py`
### Structure of deep learning model of retrieving COT and CER simultaneously
![model_structure](https://user-images.githubusercontent.com/55510330/149812967-edc4664f-59bf-46df-a4b6-c2798149a293.png)
![cer and cot retrieval structure1](https://user-images.githubusercontent.com/55510330/151222566-d017cfd2-56f0-460e-a5b6-1436aa0c12b1.png)
### Models Testing
After training any of the RNN-based deep learning model, you may run `3D_cloud_retrieval/COT retrieval_Dataset1&2/test.py` for testing. The predicted COT retrieval will be saved. You can also visualize the comparison among original COT, deep learning retrieved COT and 1D retrieval results. 
## Required Packages
* tf-nightly 2.7.0.dev20210801
* keras-nightly 2.7.0.dev2021081000   
* scikit-learn
* matplotlib 3.3.4            
* numpy 1.19.2           
* h5py 3.1.0           

### Install
Before running the codes, please install the required packages listed above:
`pip install -r requirements.txt`

## Contributors
* Xiangyang Meng, Department of Information Systems, University of Maryland Baltimore County, <xmeng1@umbc.edu>
* Adeleke Segun Ademakinwa, Department of Physics, University of Maryland Baltimore County, <adeleka1@umbc.edu>
* Zhibo Zhang, Department of Physics, University of Maryland Baltimore County, <zzbatmos@umbc.edu>
* Sanjay Purushotham, Department of Information Systems, University of Maryland Baltimore County, <psanjay@umbc.edu>

## License
  Licensed under the [MIT LICENSE](LICENSE)
