# Machine Learning for 3D Cloud Retrievals
In this GitHub repositary, you will find information on how to preprocess the raw data, how to install required packages, all the deep learning models, how to train and test the models, and how to evaluate the test results quantitively and through visualization.

## Abstract
Retrieving cloud microphysics or optical properties  from reflectance is an inverse problem.  Traditional physics-based method uses 1D retrieval to retrieve cloud properties based on the cloud’s three-dimensional (3D) radiative transfer effects. But this method suffers from significant gap between retrieved cloud properties and real cloud properties since 3D radiative-transfer effects affect the radiance values and cloud properties for cloud retrievals.  Recent researches have shown the feasibility of using deep learning models, such as convolutional neural networks and deep feed-forward neural networks, to reduce  retrieval  errors  for  multi-pixel cloud retrieval. In this study, we present various RNN-based deep learning models for cloud properties retrieval. Bi-directional Long Short Term Memory with Transformer and Embedding achieves the best outcome in retrieving Cloud Optical Thickness(COT) through reflectance values, with a distinct improvement from other deep learning algorithms.


## Datasets
Two versions of datasets summerized in Table 1:

* Version 1: simple fractal cloud for Cloud Optical Thickness (COT) only with constant cloud top 

11 Features: Solar Azimuth Angle, Solar Zenith Angle, Surface Albedo, View Zenith Angle, Liquid water path, Effective radius, Wavelength, reflectance values, Cloud Optical Thickness, spatial x, spatial y

* Version 2: simple fractal cloud for Cloud Optical Thickness (COT) only with varying cloud top

13 Features: Solar Azimuth Angle, Solar Zenith Angle, Surface Albedo, View Zenith Angle, Liquid water path, Effective radius, Wavelength, reflectance values, Cloud Optical Thickness, spatial x, spatial y, cloud_top_height(km), cloud_top_height(meters).


![datasets](https://user-images.githubusercontent.com/55510330/151377543-98ac3fea-04b9-41d2-8d7e-497a7f3a0324.png)

We use Dataset 1 and Dataset 2 for Cloud Optical Thickness(COT) retrieval: 

![WeChat Image_20211025153436](https://user-images.githubusercontent.com/55510330/138758886-be31f8ea-d4fd-42da-ac03-eb717bc92703.png)
## Pipeline
![pipeline](https://user-images.githubusercontent.com/55510330/149815510-5dbae0b2-6530-47c4-b597-6e27546f22d4.png)
### Dataset Preparation
As shown in dataset description above, each dataset is consisted of synthetic cloud profiles, and each synthetic cloud profile contains radiance values and COT values. For example, each demo cloud profile here has 4096 radiance values (input for deep learning models) and 4096 COT values （target output for deep learning models). The current resolution is 10m. We average reflectance data to 500m resolution and 1000m resolution before feeding them to the deep learning models. As a result, there are 82 (4096/50≈82)radiance values and 41(4096/100≈41) radiance values in each synthetic cloud profile, respectfully. The dataset preprocessing code prepares the dataset which will be used for the following deep learning models training and testing. For dataset preprocessing code, please refer to `DL_3d_cloud_retrieval/COT_retrieval/dataset_preparation.py`. 

### Models Training

The deep learning models for retrieving Cloud Optical Thickness are as follows. To train model:
```
python main.py --model=bilstm_transformer_embedding --radiance_file_name=data_reflectance.h5 --cot_file_name=data_cot.h5
```

A. DNN-based model (Okamura et al.) `DL_3d_cloud_retrieval/COT_retrieval/dnn.py`

B. CNN-based model (Angelof et al.) `DL_3d_cloud_retrieval/COT_retrieval/cnn.py`

C. Our proposed RNN-based models:
* BiLSTM with Transformer and Embedding `DL_3d_cloud_retrieval/COT_retrieval/bilstm_transformer_embedding.py`
### Structure of deep learning model of retreiving COT: BiLSTM with Transformer and Embedding
![BiLSTM transformer embedding (1)](https://user-images.githubusercontent.com/55510330/151221487-98c05139-8bb8-4af9-9158-155f415e9f00.png)
* BiLSTM with Transformer `DL_3d_cloud_retrieval/COT_retrieval/bilstm_with_transformer.py`
* Transformer `DL_3d_cloud_retrieval/COT_retrieval/transformer.py`
* LSTM with Transformer `DL_3d_cloud_retrieval/COT_retrieval/lstm_with_transformer.py`
* BiLSTM with Embedding `DL_3d_cloud_retrieval/COT_retrieval/bilstm_embedding.py`
* LSTM with Embedding `DL_3d_cloud_retrieval/COT_retrieval/lstm_embedding.py`
* BiLSTM `DL_3d_cloud_retrieval/COT_retrieval/bilstm.py`
* LSTM `DL_3d_cloud_retrieval/COT_retrieval/lstm.py`

Bi-directional Long Short Term Memory with Transformer and Embedding achieves the best outcome in terms of COT retrieval. Hyperparameters fine-tuning is not included in these models as this process was done off-line. 

### Models Testing
After training any of the RNN-based deep learning model, you may run `DL_3d_cloud_retrieval/COT_retrieval/test.py` for testing. The predicted COT retrieval will be saved. You can also visualize the comparison among original COT, deep learning retrieved COT and 1D retrieval results. 

To test trained model by using the original data:
```
python test.py --cot_file_name=data_cot.h5 --path_1d_retrieval=retrieved_COT/ --path_model=saved_model/bilstm_transformer_embedding/model(1).h5 --path_predictions=saved_model/bilstm_transformer_embedding/ --radiance_test=X_test_1.npy --cot_test=y_test_1.npy --path_plots=plots/bilstm_with_transformer_embedding/
```
Note you can supply: the path of a saved model you want to run into directory --path_model; the path of 1D retrieval results into directory --path_1d_retrieval. The retrieved results will be saved at the specified directory --path_predictions, and visualization results will be saved at the specified directory --path_plots.

To test trained model by using the example data provided in this GitHub repositary:
```
python test_example.py --cot_file_name=data_cot.h5 --path_1d_retrieval=retrieved_COT/ --path_model=saved_model/lstm/model(1).h5 --path_predictions=saved_model/lstm/ --radiance_test=X_test_1.npy --cot_test=y_test_1.npy --path_plots=plots/lstm/
```

## Experiments results
![c68440eee1e9637798007125c35a05d](https://user-images.githubusercontent.com/55510330/151385336-87c770d8-04ef-4fd5-8527-13240829e15b.png)
## Required Packages
* python 3.7
* tf-nightly 
* scikit-learn
* matplotlib 3.3.4            
* numpy 1.20           
* h5py 3.1.0
* cudnn 8.2.1          

### Install
```
conda create --name dl_3d_cloud_retrieval_env python=3.7
conda activate dl_3d_cloud_retrieval_env
git clone https://github.com/AI-4-atmosphere-remote-sensing/DL_3d_cloud_retrieval
cd DL_3d_cloud_retrieval
pip install -r requirements.txt
conda install cudnn
```

## Contributors
* Xiangyang Meng, Department of Information Systems, University of Maryland Baltimore County, <xmeng1@umbc.edu>
* Adeleke Segun Ademakinwa, Department of Physics, University of Maryland Baltimore County, <adeleka1@umbc.edu>
* Zhibo Zhang, Department of Physics, University of Maryland Baltimore County, <zzbatmos@umbc.edu>
* Sanjay Purushotham, Department of Information Systems, University of Maryland Baltimore County, <psanjay@umbc.edu>

## References
Angelof, K., Bergstrom, K., Le, T., Xu, C., Rajapakshe, C., Zheng, J., & Zhang, Z. (2020). Machine Learning for Retrieving Cloud Optical Thickness from Observed Reflectance: 3D Effects CyberTraining: Big Data+ High-Performance Computing+ Atmospheric Sciences. UMBC Faculty Collection.

Okamura, R., Iwabuchi, H., & Schmidt, K. S. (2017). Feasibility study of multi-pixel retrieval of optical thickness and droplet effective radius of inhomogeneous clouds using deep learning. Atmospheric Measurement Techniques, 10(12), 4747-4759.

## License
  Licensed under the [MIT LICENSE](LICENSE)
