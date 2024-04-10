
# Deep Learning for 3D Cloud Retrievals from LES cloud scences
In this GitHub repositary, you will find information on how to preprocess the raw data, how to install required packages, all the deep learning models, how to train and test the models, and how to evaluate the test results quantitively and through visualization.


## Create conda environment
```
conda create -n cot_retrieval python=3.9 
conda activate cot_retrieval
```
### Install Dependencies
* Install Pytorch: ```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```
* Install Torchinfo (1.7.1): ```pip install torchinfo```
* Install matplotlib (3.6.1): ```pip install matplotlib```
* Install mmcv (1.6.2): ```pip install mmcv``` [If you face error regarding mmcv when running, try ```pip install mmcv-full``` ]
* Install scikit-learn (): ```pip install scikit-learn``` [please ignore the following error message: ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. nltk 3.7 requires click, which is not installed. nltk 3.7 requires tqdm, which is not installed. ]
* Install h5py: ```pip install h5py```
* Install wandb: ```pip install wandb``` [We have used wandb to profile the training, please refer to https://docs.wandb.ai/ for more information.]



## Data Preparation
The following bash script processes the raw data which is in hdf format into numpy arrays. It also creates a five fold cross validated dataset. The scripts requires the dataset path.

For data preprocessing, run the following command.
```
bash data_preprocess.sh $data_path
```



### Models Training
After data preprocessing, one can start training deep learning model. Run the following bash script to train the model.
```
bash train.sh $gpu/cpu $model_arch 
```
* $gpu/cpu == Write [cpu] if no GPU available. Otherwise write the GPU number. e.g., [0]
* $model_arch == Specify the model architecture. e.g., [cloudunet]

### Models Testing
Download the pretrained model from the following URL.
```
https://drive.google.com/drive/folders/1_vLSWM-tCpAFwfGfZaJfLhOJayefwE-9?usp=sharing
```
Create directories here and place the model inside the directoy. It should look like this.
* saved_model/cloudunet/cloudunet.pth
* saved_model/okamura/okamura.pth

To test trained model by using the example data provided in this GitHub repositary:
```
python test.py --model_arch cloudunet --model_filename cloudunet.pth 
```
Set the model_arch name to [okamura] and model_filename to [okamura.pth] to test the okamura model.

## Contributors
* Zahid Hassan Tushar, Department of Information Systems, University of Maryland Baltimore County, <ztushar1@umbc.edu>
* Adeleke Segun Ademakinwa, Department of Physics, University of Maryland Baltimore County, <adeleka1@umbc.edu>
* Zhibo Zhang, Department of Physics, University of Maryland Baltimore County, <zzbatmos@umbc.edu>
* Sanjay Purushotham, Department of Information Systems, University of Maryland Baltimore County, <psanjay@umbc.edu>

## References
Okamura, R., Iwabuchi, H., & Schmidt, K. S. (2017). Feasibility study of multi-pixel retrieval of optical thickness and droplet effective radius of inhomogeneous clouds using deep learning. Atmospheric Measurement Techniques, 10(12), 4747-4759.

## License
  Licensed under the [MIT LICENSE](LICENSE)
