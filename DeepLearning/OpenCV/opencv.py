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
random_image = np.random.randint(0,256, size=(200,200,3))
print(random_image.shape)
# %%
outpath = './random_image.png'
cv2.imwrite(outpath,random_image)
# %%
my_img = cv2.imread('./random_image.png')
# %%
print(type(my_img))
print(my_img.shape)
# %%
cv2.imshow('', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 랜덤 그림 !!!
img = np.zeros((512,512,3),np.uint8)
# %%
plt.imshow(img)
plt.show()
# %% 선긋기!!!
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
plt.imshow(img)
plt.show()
# %% 사각형 그리기!!!
img = cv2.rectangle(img,(400,0),(510,128),(0,255,0),3)
plt.imshow(img)
plt.show()
# %% 원 그리기!!!
img = cv2.circle(img,(450,50),50,(0,0,255),-1)
plt.imshow(img)
plt.show()
# %%
img = cv2.circle(img,(50,450),50,(0,255,255),2)
plt.imshow(img)
plt.show()
# %% 타원 그리기!!!
img = cv2.ellipse(img,(256,256),(150,30),0,0,180,(0,255,0),-1)
plt.imshow(img)
plt.show()
# %%
img = cv2.ellipse(img,(256,256),(150,30),45,0,360,(255,255,255),2)
plt.imshow(img)
plt.show()
# %%
img = cv2.ellipse(img,(256,256),(150,10),135,0,270,(0,0,255),2)
plt.imshow(img)
plt.show()
# %%다각형 그리기!!!
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
print(pts.shape)
# %%
pts = pts.reshape(-1,2,1)
print(pts.shape)
img = cv2.polylines(img,[pts],True,(0,150,250),4)
# %%
plt.imshow(img)
plt.show()
# %%
pts2 = np.array([[150,5],[200,30],[100,70],[50,200]],np.int32)
print(pts2.shape)
#%%
pts2 = pts2.reshape(-1,1,2)
print(pts2.shape)
img=cv2.polylines(img,[pts2],True,(172,200,255),4)
# %%
plt.imshow(img)
plt.show()
# %%텍스트 그리기!!!
img = cv2.putText(img,'OpenCV',(10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),3)
# %%
plt.imshow(img)
plt.show()
# %% 컬러 매핑!!!@@@!!!
