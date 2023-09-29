#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import requests
from io import BytesIO
# %%
url = 'https://cdn.pixabay.com/photo/2017/09/25/13/12/cocker-spaniel-2785074_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('dog.jpg')
# %%
orgin_img = cv2.imread('./dog.jpg')
print(orgin_img.shape)
# %%
plt.imshow(orgin_img)
plt.show()
# %%
img_rgb = cv2.cvtColor(orgin_img,cv2.COLOR_BGR2RGB)

# %%
plt.imshow(img_rgb)
plt.show()
# %%
img_hsv = cv2.cvtColor(orgin_img,cv2.COLOR_BGR2HSV)
plt.imshow(img_hsv)
plt.show()
# %%
print(img_hsv)
# %%
print(np.max(img_hsv),np.min(img_hsv))

# %%
print(np.max(img_hsv[:,0,0]))
print(np.min(img_hsv[:,0,0]))
# %%
img_hsl = cv2.cvtColor(orgin_img,cv2.COLOR_BGR2HLS)
plt.imshow(img_hsl)
plt.show()
# %%
print(img_hsl)
# %%
print(np.max(img_hsl),np.min(img_hsl))

# %%
print(np.max(img_hsl[:,0,0]))
print(np.min(img_hsl[:,0,0]))
# %%
img_ycrcb = cv2.cvtColor(orgin_img,cv2.COLOR_BGR2YCrCb)
plt.imshow(img_ycrcb)
plt.show()
# %%
print(img_ycrcb)
# %%
print(np.max(img_ycrcb),np.min(img_ycrcb))
# %%
img_gray = cv2.cvtColor(orgin_img,cv2.COLOR_BGR2GRAY)
print(img_gray.shape)
# %%
plt.imshow(img_gray,cmap='gray')
plt.show()
# %%
print(img_gray)
# %%
print(np.max(img_gray),np.min(img_gray))

# %%
url = 'https://cdn.pixabay.com/photo/2015/10/09/00/55/lotus-978659_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('flower1.jpg')

#%%
url = 'https://cdn.pixabay.com/photo/2012/03/01/00/55/garden-19830_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('flower2.jpg')
# %%
img1 = cv2.imread('./flower1.jpg',0)
img2 = cv2.imread('./flower2.jpg',0)
# %%
hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
# %%
plt.figure(figsize=(12,8))
plt.subplot(221) #2/2 칸의 1번째
plt.imshow(img1,'gray')
plt.title('Flower1')

plt.subplot(222)
plt.imshow(img2,'gray')
plt.title('Flower2')

plt.subplot(223)
plt.plot(hist1,color='r')
plt.plot(hist2,color='g')
plt.xlim([0,256])
plt.title('Histogram')

plt.show()
# %%
url = 'https://cdn.pixabay.com/photo/2020/03/12/04/07/cat-4923824_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('cat.jpg')
# %%
img = cv2.imread('./cat.jpg')
print(img.shape)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# %%
mask = np.zeros(img.shape[:2],np.uint8) #흑색
mask[180:400,260:600] = 255 # 지정 영역만 흰색
# %%
masked_img = cv2.bitwise_and(img,img,mask=mask)
# %%
hist_full = cv2.calcHist([img],[1],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[1],mask,[256],[0,256])
# %%
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(img,'gray')
plt.title('Original Image')

plt.subplot(222)
plt.imshow(mask,'gray')
plt.title('Mask')

plt.subplot(223)
plt.imshow(masked_img)
plt.title('Mask Image')


plt.subplot(224)
plt.title('Histogram')
plt.plot(hist_full,color='r')
plt.plot(hist_mask,color='b')
plt.xlim([0,256]) # x축 길이

plt.show()
# %% 균일화 !!!
url = 'https://cdn.pixabay.com/photo/2020/08/03/13/33/taiwan-5460063_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('taiwan.jpg')

# %% 평탄화 작업
img = cv2.imread('./taiwan.jpg',0) #차원없이
print(img.shape)
# %%
hist,bins = np.histogram(img.flatten(),256,[0,256])
# %%
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf,0) # cdf가 0인부분 mask 처리해줘
# %%
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
# %%
img2 = cdf[img]

plt.figure(figsize=(10,8))
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(img2,'gray')
plt.title('Equalization')

plt.show()
# %%
img = cv2.imread('./taiwan.jpg',0) #차원없이
print(img.shape)
# %% cv2 함수 사용해 평탄화
img2 = cv2.equalizeHist(img)
# %%
plt.figure(figsize=(10,8))
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(img2,'gray')
plt.title('Equalization')

plt.show()
# %%
url = 'https://cdn.pixabay.com/photo/2015/08/13/01/00/keyboard-886462_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('keyboard.jpg')

# %%
img = cv2.imread('./keyboard.jpg',0) #차원없이
print(img.shape)
# %%
img2 = cv2.equalizeHist(img)
# %%
plt.figure(figsize=(10,8))
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(img2,'gray')
plt.title('Equalization')

plt.show()
# %%
img = cv2.imread('./keyboard.jpg',0)
print(img.shape)
# %%
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
img2 = clahe.apply(img)
# %%
plt.figure(figsize=(10,8))
plt.subplot(121)
plt.imshow(img,'gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(img2,'gray')
plt.title('Equalization')

plt.show()
# %%
