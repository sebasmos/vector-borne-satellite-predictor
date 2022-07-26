import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import os
import time
import sys
sys.path.insert(0,'..')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import  mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import signal
import pickle

from sklearn.decomposition import PCA

from epiweeks import Week, Year
from datetime import date

from datetime import date

from random import randint, randrange
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

import skimage
import cv2
import os
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt

import cv2 
from PIL import Image
from imutils import paths
import argparse
import time
import sys
import cv2
import os
def get_MAE_score(y_test, y_pred):
  # y_test = y_test.detach().numpy()
  # y_pred = y_pred.detach().numpy()
  # y_test = torch.from_numpy(y_test)
  # y_pred = torch.from_numpy(y_pred)
  return round(mean_absolute_error(y_test, y_pred), 4)

def get_MAPE_score(y_true, y_pred):
  """Get Mean Absolute Percentage Error (MAPE)
  
  Calculate the MAPE score based on the prediction. 
  The lower MAPE socre is, the better the predictions are.

  """
  return round(mean_absolute_percentage_error(y_true, y_pred), 4)
def readImg(img_path, resize_ratio=None):
  img = io.imread(img_path)

  if resize_ratio:
    img_rescale = rescale(img, resize_ratio, anti_aliasing=True)

  print(os.path.basename(img_path), '(origin shape:', img.shape, '-> rescale:', str(img_rescale.shape) + ')')
  return img_rescale


# Load data from one of the source
def loadData(csv_folder, img_folder, option=None, resize_ratio=None):
  if option is None:
    # Get data by combining from csv and images
    df = loadStructuredData(csv_folder)
    info_dict = combineData(img_folder, df, resize_ratio)
    
    print(len(info_dict['LastDayWeek']), len(info_dict['Image']), len(info_dict['cases_medellin']))

  else:
    # Load data from previous pickle file
    info_dict = 1#loadDataFromPickle(option)
  return info_dict
  

def loadStructuredData(csv_path):
  df = pd.DataFrame()
  if os.path.isdir(csv_path):
    for filename in os.listdir(csv_path):
      file_path = os.path.join(csv_path, filename)
      df = df.append(pd.read_csv(file_path))
  elif os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
  else:
    print('Error: Not folder or file')
  return df
  
def getEpiWeek(origin_str):
  """Get epi week from string
  """
  date_ls = origin_str.split('-')
  return Week.fromdate(date(int(date_ls[0]), int(date_ls[1]), int(date_ls[2])))
  
def combineData(img_folder, df, resize_ratio=None):
  info_dict = {'LastDayWeek':[], 'cases_medellin':[], 'Image':[], 'epi_week':[]}
  img_list = os.listdir(img_folder)

  for index, row in df.iterrows():
    name = row['LastDayWeek']
    week_df = str(getEpiWeek(name))
    case = row['cases_medellin']
    for img_name in img_list:
      
      # If image name is image_2017-12-24.tiff -> get 2017-12-24
      # Reference Links: https://www.w3schools.com/python/ref_string_join.asp, 
      #                  https://stackoverflow.com/questions/13174468/how-do-you-join-all-items-in-a-list/13175535
      new_img_name = ''.join(i for i in img_name if i.isdigit() or i == '-')      

      week_img = str(getEpiWeek(new_img_name))
      #print(f"{week_df} = {week_img}")
      if week_df == week_img:
        #print("ENTRO")
        img_path = os.path.join(img_folder, img_name)
        img = readImg(img_path, resize_ratio)

        info_dict['Image'].append(img[:,:,1:4])
        info_dict['LastDayWeek'].append(name)
        info_dict['cases_medellin'].append(case)
        info_dict['epi_week'].append(week_df)
        break

  return info_dict

def splitTrainTestSet(ratio):
  # Split the data into training (ratio) and testing (1 - ratio)
  train_val_ratio = ratio
  train_num = int(len(info_dict['Image']) * train_val_ratio)

  # Change list to array
  origin_dimension_X = np.array(info_dict['Image'])
  labels = np.array(info_dict['cases_medellin'])

  print(''.center(60,'-'))

  origin_X_train = origin_dimension_X[:train_num,:,:,:]
  y_train = labels[:train_num]
  origin_X_test = origin_dimension_X[train_num:,:,:,:]
  y_test = labels[train_num:]

  # print('Total number of weeks:'.ljust(30), len(origin_dimension_X), 'weeks')
  # print('Training input:'.ljust(30), origin_X_train.shape)
  # print('Training output:'.ljust(30), y_train.shape)
  # print('Testing input:'.ljust(30), origin_X_test.shape)
  # print('Testing output:'.ljust(30), y_test.shape) 

  return origin_X_train, y_train, origin_X_test, y_test

# Polynomial Regression
def calc_r_2(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])

    return ssreg / sstot

def classified_with_SVR(origin_X_train, origin_X_test, y_train, y_test):
  print('[SVR]'.center(100, '-'))

  reshape_X_train = origin_X_train.reshape(origin_X_train.shape[0], -1)
  reshape_X_test = origin_X_test.reshape(origin_X_test.shape[0], -1)

  regressor = SVR(C=1.0, epsilon=0.2)
  regressor.fit(reshape_X_train, y_train)

  float_y_pred = regressor.predict(reshape_X_test)
  int_y_pred = [int(i) for i in float_y_pred]

  print('Predicted')
  print(' '.ljust(3, ' '), 'List =', int_y_pred)
  print(' '.ljust(3, ' '), 'Mean =', round(np.mean(int_y_pred), 4))
  print('')

  print('Real')
  print(' '.ljust(3, ' '), 'List =', y_test)
  print(' '.ljust(3, ' '), 'Mean =', round(np.mean(y_test), 4))
  print('')
  
  MAE = get_MAE_score(y_test, int_y_pred)
  MAPE = get_MAPE_score(y_test, int_y_pred)

  r_2 = calc_r_2(y_test, int_y_pred, 15)

  print('- MAE: ', str(MAE).rjust(8), '(cases different in average)')
  print('- MAPE:', str(MAPE).rjust(8), '(times different in average)')
  print('- r_squared:', str(r_2).rjust(8), '(times different in average)')

  return MAE, MAPE, r_2



def dimension_reduct_with_PCA(origin_X_train, origin_X_test, y_train):
  print(' PRINCIPAL COMPONENT ANALYSIS  '.center(100, '='))

  reshape_X_train = origin_X_train.reshape(origin_X_train.shape[0], -1)
  reshape_X_test = origin_X_test.reshape(origin_X_test.shape[0], -1)

  pca = PCA(n_components=0.95) 
  pca_X_train = pca.fit_transform(reshape_X_train)

  pca_X_test = pca.transform(reshape_X_test)
  print('Origin shape'.ljust(15), reshape_X_train.shape)
  print('Resize shape'.ljust(15), pca_X_train.shape)  

  return pca_X_train, pca_X_test

from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.y[ind]
        return x, y
    
class TestDataset(TrainDataset):
    def __getitem__(self, ind):
        x = self.data[ind]
        return x
    
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.datasets.utils import download_file_from_google_drive
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    img_folder = "./data/top5/DATASET_5_best_cities/Medellin"
    csv_folder = "./dengue/merge_cases_temperature_WeeklyPrecipitation_timeseries.csv"
    info_dict = loadData(csv_folder, img_folder, resize_ratio=(0.7, 0.7, 1))#gpeg
        
    print('INFO_DICT'.center(50, '-'))
    print('keys:', info_dict.keys())
    print('')
    
    print('DENGUE CASES'.center(50, '-'))
    print('Max weekly dengue cases:', max(info_dict['cases_medellin']))
    print('Min weekly dengue cases:', min(info_dict['cases_medellin']))
    print('')
    
    print('WEEKS'.center(50, '-'))
    print('Max week:', max(info_dict['LastDayWeek']))
    print('Min week:', min(info_dict['LastDayWeek']))
    train_val_ratio = 0.8
    train_num = int(len(info_dict['Image']) * train_val_ratio)
    
      # Change list to array
    origin_dimension_X = np.array(info_dict['Image'])
    labels = np.array(info_dict['cases_medellin'])
    train_val_ratio = 0.8
    
    train_num = int(len(info_dict['Image']) * train_val_ratio)
    
      # Change list to array
    origin_dimension_X = np.array(info_dict['Image'])
    labels = np.array(info_dict['cases_medellin'])
    
    print(''.center(60,'-'))
    
    origin_X_train = origin_dimension_X[:train_num,:,:,:]
    y_train = labels[:train_num]
    origin_X_test = origin_dimension_X[train_num:,:,:,:]
    y_test = labels[train_num:]
    
    print(f"origin_X_train: {origin_X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"origin_X_train: {origin_X_test.shape}")
    print(f"y_train: {y_test.shape}")

    train_set = TrainDataset(origin_dimension_X, y_train)
    test_set  = TrainDataset(origin_X_test, y_test)

    batch_size = 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model = models.resnet50(pretrained=True)
    model.to(device)    
    

if __name__ == "__main__":
    main()