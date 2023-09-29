#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import requests
from io import BytesIO
# %%
url = 'https://cdn.pixabay.com/photo/2018/10/01/09/21/pets-3715733_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
# %%
pic
# %%
type(pic) #jpeg 형식
# %%
pic_arr = np.array(pic)
# %%
type(pic_arr)
# %%
pic_arr.shape # 높이, 너비, 3차원
# %%
pic_arr
# %%
plt.imshow(pic_arr)
plt.show()
# %%
#RGB에 따른 이미지 확인
pic_copy = pic_arr.copy()
plt.imshow(pic_copy)
plt.show()
# %%
pic_copy.shape
# %%
#채널순서(RGB:012)
print(pic_copy[:,:,0]) # RGB중 R채널만 보여줘 : 
print(pic_copy[:,:,0].shape)
# %%
plt.imshow(pic_copy[:,:,0]) # 
plt.show()
# %%
plt.imshow(pic_copy[:,:,0],cmap='gray')
plt.show() # 레드 체널에 대한 빛 정보 그레이 채널
# %%
print(pic_copy[:,:,1])
print(pic_copy[:,:,1].shape)
# %%
plt.imshow(pic_copy[:,:,1],cmap='gray')
plt.show() # 그린 채널에 대한 빛정보 그레이 채널
# %%
print(pic_copy[:,:,2])
print(pic_copy[:,:,2].shape)
# %%
plt.imshow(pic_copy[:,:,2],cmap='gray')
plt.show() #블루 채널에 대한 빛정보 그레이 채널
# %%
