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


### Test on single example
To test trained model by using the example data provided in this GitHub repositary:
```
python example_test.py --config configs/config_test_example.yaml
```
## Contributors
* Zahid Hassan Tushar, Department of Information Systems, University of Maryland Baltimore County, <ztushar1@umbc.edu>
* Adeleke Segun Ademakinwa, Department of Physics, University of Maryland Baltimore County, <adeleka1@umbc.edu>
* Zhibo Zhang, Department of Physics, University of Maryland Baltimore County, <zzbatmos@umbc.edu>
* Sanjay Purushotham, Department of Information Systems, University of Maryland Baltimore County, <psanjay@umbc.edu>

## References
Zahid Hassan Tushar, Adeleke Ademakinwa, Jianwu Wang, Zhibo Zhang, and Sanjay Purushotham. "Joint cloud optical thickness and cloud effective radius property retrievals using attention-based deep learning models". In AGU Annual Meeting. American Geophysical Union (AGU), 2024.

## License
  Licensed under the [MIT LICENSE](LICENSE)
