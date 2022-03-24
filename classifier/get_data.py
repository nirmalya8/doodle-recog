'''
This file is used to get the data from the cloud in the form of
.npy files, which can later be loaded for training the machine
learning model. It just hits the API and downloads the .npy file
into the Data folder for each of the classes.
'''
from urllib import request
def get_dataset():
  
  base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  classes = ['soccer_ball','square','sun','t-shirt','pizza','face','axe','book','apple','calculator']
  for idx,c in enumerate(classes):
    print(f"Downloading {c} ..... {idx+1}/{len(classes)}")
    class_url = c.replace('_', '%20')
    path = base_url+class_url+'.npy'
    print(path)
    request.urlretrieve(path, 'Data/'+c+'.npy')
    print(f"Downloaded {c}")
    print("==========================================")

if __name__=='__main__':
    get_dataset()