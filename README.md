# Kaggle_Sea-Lions-Counting


This is our solution on [NOAA Fisheries Steller Sea Lion Population Count](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count)<br>
Based on [@outrunner - The 1st-place winner in the competition](https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions)

## Hardware
The following specs were used to create the solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
- 1x NVIDIA TitanXp 


## Installation
All requirements should be detailed in [requirements.txt](https://github.com/yan-roo/Mask_RCNN-TinyVOC/blob/master/requirements.txt). Using Anaconda is strongly recommended.
```
$ conda create -n sealions python=3.6
$ source activate sealions
$ pip install -r requirements.txt
```

## Dataset Preparation
Download the data from [Kaggle](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/data)
or Use Kaggle Api
```
$ kaggle competitions download -c noaa-fisheries-steller-sea-lion-population-count
```
Unzip with the key (kaggle2017steller)
```
$ 7z x KaggleNOAASeaLions.7z
```


### Crop and Divided Images into Training/Validation set
After unzip the 7z file, the data directory is structured as:
```
Kaggle_Sea-Lions-Counting
  +- TrainDotted
  +- Train
  +- Test
  +- data_password.txt
  +- MismatchedTrainImages.txt
  +- csv
  +- patch-image-csv.ipynb
  +- submit.py
  +- use-keras-to-count-sea-lions.ipynb
  ```
  
Create new directories and use [patch-image-csv](https://github.com/yan-roo/Kaggle_Sea-Lions-Counting/blob/master/patch-image-csv.ipynb)to create cropped images
```
$ mkdir 300x300
$ cd 300x300/
$ mkdir sea_lion
$ mkdir background
```

```
  +- 300x300
  |  +- sea_lion
  |  +- background
```



## Training
Following the Jupyter Notebook [use-keras-to-count-sea-lions](https://github.com/yan-roo/Kaggle_Sea-Lions-Counting/blob/master/use-keras-to-count-sea-lions.ipynb). And you will get your training weights in logs directory.


### Make Submission use [submit.py](https://github.com/yan-roo/Kaggle_Sea-Lions-Counting/blob/master/submit.py)
```
$ python submit.py
```
Use Kaggle API to submit result
```
$ kaggle competitions submit -c noaa-fisheries-steller-sea-lion-population-count -f submission.csv -m "Message"
```
