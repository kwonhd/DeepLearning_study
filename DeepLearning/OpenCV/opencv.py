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
# R 채널 분포
pic_red = pic_arr.copy()
pic_red[:,:,1] = 0
pic_red[:,:,2] = 0
# %%
pic_red
# %%
plt.imshow(pic_red)
plt.show()
# %%
pic_green = pic_arr.copy()
pic_green[:,:,0] = 0
pic_green[:,:,2] = 0
#%%
pic_green
# %%
plt.imshow(pic_green)
plt.show()
# %%
pic_blue = pic_arr.copy()
pic_blue[:,:,0] = 0
pic_blue[:,:,1] = 0
#%%
pic_blue
# %%
plt.imshow(pic_blue)
plt.show()
# %%
# 이미지 출력 (opencv)
# 이미지를 창에 표시합니다

cv2.imshow('', pic_arr)
# 키 입력을 대기하고 그 다음 창을 닫습니다
cv2.waitKey(0)
cv2.destroyAllWindows()# %%
#opencv 채널순서는 BGR 순서이다!!

# %%
image = cv2.cvtColor(pic_arr, cv2.COLOR_RGB2BGR) # RGB로 변경된 색
cv2.imshow('',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
print(image[0][0]) # BGR
# %%
print(pic_arr[0][0]) # RGB
# %%
url = 'https://cdn.pixabay.com/photo/2018/10/01/09/21/pets-3715733_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('dog.jpg') # 현재 수행중인 가상환경에 사진 저장
# %%
cv_image = cv2.imread('dog.jpg', cv2.IMREAD_UNCHANGED)
# %%
print(type(cv_image))
# %%
print(cv_image)
# %%
cv2.imshow('', cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
plt.imshow(cv_image)
plt.show()
# %%
image_temp = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
# %%
plt.imshow(image_temp)
plt.show()
# %%
url = 'https://cdn.pixabay.com/photo/2017/01/12/21/42/tiger-1975790_1280.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('tiger.jpg')
# %%
img_gray = cv2.imread('tiger.jpg',cv2.IMREAD_GRAYSCALE)
# %%
print(img_gray.shape) # 2차배열 그레이채널
# %%
cv2.imshow('', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
plt.imshow(img_gray)
plt.show()
# %%
plt.imshow(img_gray, cmap='gray')
plt.show()
# %%
plt.imshow(img_gray, cmap='magma')
plt.show()

# %%
### 이미지 쓰기
