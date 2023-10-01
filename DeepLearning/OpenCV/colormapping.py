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
url = 'https://cdn.pixabay.com/photo/2020/08/14/15/22/canal-5488271_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('canal.jpg')
# %%
img = cv2.imread('./canal.jpg')
print(img.shape)
# %%
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# %%
hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
# %%
plt.imshow(hist) # 파란색 노란색이 주를 이룸
plt.show()
# %%
url = 'https://cdn.pixabay.com/photo/2021/06/26/06/52/moon-6365467_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('moon.jpg')
# %%
img = cv2.imread('./moon.jpg')
print(img.shape)
#%%
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
height, width = img.shape[:2]

# %%
shrink = cv2.resize(img, None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA) # 줄이기
zoom = cv2.resize(img,(width*2,height*2),interpolation=cv2.INTER_CUBIC) # 직접 키우기
zoom2 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) # 비율로 키우기
# %%
cv2.imshow('', shrink)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(shrink.shape)
# %%
cv2.imshow('', zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(zoom.shape)
# %%
cv2.imshow('', zoom2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(zoom2.shape)
# %% 이미지 위치 변경
rows, cols = img.shape[:2]

M = np.float32([[1,0,20],[0,1,40]]) #x축으로 20, y축으로 40 이동[]로 묶어야함
dst = cv2.warpAffine(img,M,(cols,rows))
#%%
cv2.imshow('', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
print(img.shape)
# %%
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2,rows/2),60,0.5) #시계반대 60도 회전, 크기 50%감소
dst = cv2.warpAffine(img,M,(cols,rows))
# %%
cv2.imshow('', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(dst.shape)
# %%
img = cv2.imread('./moon.jpg')
print(img.shape)
# %%
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# %%
plt.imshow(img)
plt.show()
# %%
result1 = cv2.flip(img,1) # 좌우반전
# %%
plt.imshow(result1)
plt.show()
# %%
result2 = cv2.flip(img,0) # 0 = 상하반전
plt.imshow(result2)
plt.show()
# %%
result3 = cv2.flip(img,-1) # 0 = 상하좌우반전
plt.imshow(result3)
plt.show()
# %%
rows,cols, ch = img.shape

pts1 = np.float32([[200,100],[400,100],[200,200]])
pts2 = np.float32([[200,300],[400,200],[200,400]])

cv2.circle(img,(200,100,),10,(255,0,0),-1)
cv2.circle(img,(400,100),10,(0,255,0),-1)
cv2.circle(img, (200,200),10,(0,0,255),-1)

M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
# %%
cv2.imshow('', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(dst.shape)
# %%
plt.subplot(121)
plt.imshow(img[:,:,::-1])
plt.title('image')

plt.subplot(122)
plt.imshow(dst[:,:,::-1])
plt.title('Affine')

plt.show()
# %% 원근법 변환
url = 'https://cdn.pixabay.com/photo/2015/04/04/06/54/train-706219_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('train.jpg')
# %%
road = cv2.imread('./train.jpg')
print(road.shape)
# %%
cv2.imshow('', road)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% 좌표점은 (왼쪽 위)→(오른쪽 위)→(오른쪽 아래)→(왼쪽 아래)
top_left = (180,300)
top_right = (270,300)
bottom_left = (80,550)
bottom_right = (400,550)

pts1 = np.float32([top_left,top_right,bottom_right,bottom_left])

w1 = abs(bottom_right[0]-bottom_left[0])
w2 = abs(top_right[0]-top_left[0])
h1 = abs(top_right[1]-bottom_right[1])
h2 = abs(top_left[1]-bottom_left[1])

max_width = max([w1,w2])
max_height = max([h1,h2])

pts2 = np.float32([[0,0],
                  [max_width-1,0],
                  [max_width-1,max_height-1],
                  [0,max_height-1]])

# %%
cv2.circle(road,top_left,10,(255,0,0),-1)
cv2.circle(road,top_right,10,(0,255,0),-1)
cv2.circle(road,bottom_right,10,(0,0,255),-1)
cv2.circle(road,bottom_left,10,(255,255,255),-1)

# %%
cv2.imshow('', road)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(road,M,(max_width,max_height))
# %%
plt.subplot(121)
plt.imshow(road[:,:,::-1])
plt.title('image')

plt.subplot(122)
plt.imshow(dst[:,:,::-1])
plt.title('Perspective')

plt.show()
#위에서 평면으로 보는 것 처럼 보임
# %%
url = 'https://cdn.pixabay.com/photo/2020/08/22/00/24/desert-5507220_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('tree.jpg')
# %%
tree = cv2.imread('./tree.jpg')
print(tree.shape)
# %%
cv2.imshow('', tree)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% tree 사진의 200,200 위치의 value 
temp_px = tree[200,200]
print(temp_px)
# %%
temp_channel = tree[200,200,2]
print(temp_channel)
# %% 픽셀 값 변경
tree[100,100] = [0,0,255]
tree[101,100] = [0,0,255]
tree[102,100] = [0,0,255]
tree[103,100] = [0,0,255]
tree[104,100] = [0,0,255]
# %%
cv2.imshow('', tree)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% 인덱싱
print(tree[:100,:100].shape)

# %%
cv2.imshow('', tree[:100,:100])
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
white_box = tree[:100,:100]
white_box = [255,255,255]
tree[:100,:100] = white_box
# %%
cv2.imshow('', tree)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% 이미지 ROI
tree = cv2.imread('./tree.jpg')
print(tree.shape)
cv2.imshow('', tree)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
t = tree[290:360,430:530]
# %%
cv2.imshow('', t)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
tree[200:270,380:480] = t
cv2.imshow('', tree)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
tree[200:270,500:600] = t
cv2.imshow('', tree)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
b,g,r = cv2.split(tree)

# %%
print(b)
print(b.shape)
#%%
print(g)
print(g.shape)
#%%
print(r)
print(r.shape)
# %%
img = cv2.merge((b,g,r))
# %%
print(img.shape)
# %% R채널 0으로 값 변경
img[:,:,2] =0 
# %%
print(img)
# %% 붉은색이 빠진 사진
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% G채널 0으로 값 변경
img[:,:,1] =0 
# %%
print(img)
# %% 초록색도 빠진 사진
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% B채널 0으로 변경 / 검정화면
img[:,:,0] = 0
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
x = np.uint8([250])
y = np.uint8([10])
# %% cv2.add() 연산
print(cv2.add(x,y)) #최대값 255 넘으면 255로
# %%
print(x+y) # 260을 최대값 256으로 나눈 나머지 값
# %%
url = 'https://cdn.pixabay.com/photo/2021/01/26/17/18/cavalier-king-charles-spaniel-5952324_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('dog2.jpg')
#%%
url = 'https://cdn.pixabay.com/photo/2017/09/25/13/14/dog-2785077_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('dog3.jpg')
# %%
dog2 = cv2.imread('./dog2.jpg',cv2.IMREAD_COLOR)
print(dog2.shape)
# %%
dog3 = cv2.imread('./dog3.jpg',cv2.IMREAD_COLOR)
print(dog3.shape)
# %% add연산
result1 = cv2.add(dog2,dog3)
# %%
cv2.imshow('', result1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%넘파이 연산
result2 =dog2+dog3
# %%
cv2.imshow('', result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
url = 'https://static.vecteezy.com/system/resources/previews/000/440/202/original/star-vector-icon.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('star.jpg')

# %%
star = cv2.imread('./star.jpg',cv2.IMREAD_COLOR)
print(star.shape)
# %%
plt.imshow(star)
plt.show()
# %%
black_img = np.zeros((5120,5120,3))
# %%
plt.imshow(black_img)
plt.show()
# %%
black_img[:,:2560,:] = (255,255,255)
# %%
img = black_img.astype(np.uint8)
#%%
plt.imshow(img)
plt.show()
# %% bitwise_and 연산
result = cv2.bitwise_and(img,star)
# %%
plt.imshow(result)
plt.show()
# %% bitwise_or 연산
result2 = cv2.bitwise_or(img,star)
# %%
plt.imshow(result2)
plt.show()
# %%
result3 = cv2.bitwise_not(img)
# %%
plt.imshow(result3)
plt.show()
#%%
result4 = cv2.bitwise_not(star)
# %%
plt.imshow(result4) # 색 반전
plt.show()
# %%
result5 = cv2.bitwise_xor(star,img)
# %%
plt.imshow(result5) # T T > F // F F > T
plt.show()
# %%
temp_star = cv2.bitwise_not(star)
result6 = cv2.bitwise_xor(temp_star,img)
plt.imshow(result6)
plt.show()
#%%
url = 'https://cdn.pixabay.com/photo/2015/01/07/15/51/woman-591576_1280.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('sunset.jpg')
# %%
url = 'https://velog.velcdn.com/images/checking_pks/post/8cfd2fcc-35dd-474c-a5e4-5ac6ad3477f1/image.png'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('opencv.png')
# %%
sunset = cv2.imread('./sunset.jpg')
print(sunset.shape)
#%%
cv2.imshow('', sunset)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
opencv = cv2.imread('./opencv.png')
print(opencv.shape)
# %%
cv2.imshow('', opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
rows, cols, channels = sunset.shape
opencv = cv2.resize(opencv, (cols, rows))
# %%
opencv.shape
# %%
opencv_gray = cv2.cvtColor(opencv,cv2.COLOR_BGR2GRAY)
plt.imshow(opencv_gray, cmap='gray')
plt.show()
# %%
ret, mask = cv2.threshold(opencv_gray,200,255,cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
# %%
plt.imshow(mask, cmap='gray')
plt.show()
# %%
plt.imshow(mask_inv, cmap='gray')
plt.show()
# %%
img1_fg = cv2.bitwise_and(opencv,opencv,mask=mask) # 로고 빼고 나머지 검은색
img2_bg = cv2.bitwise_and(sunset,sunset,mask=mask_inv)
# %%
dst = cv2.add(img1_fg,img2_bg)
# %%
cv2.imshow('', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
sunset[0:rows,0:cols] = dst
# %%
cv2.imshow('', sunset)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
url = 'https://cdn.pixabay.com/photo/2013/07/13/12/42/do-not-copy-160137_1280.png'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('do-not-copy.png')
# %%
url = 'https://cdn.pixabay.com/photo/2020/06/20/12/55/fashion-5320934_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('fashion.jpg')
# %%
img1 = cv2.imread('./fashion.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread('./do-not-copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
#%%
img2 = cv2.imread('./do-not-copy.png')

# 원본 이미지를 복사합니다.
result_img = img2.copy()

# 원본 이미지에서 붉은색 글씨 부분을 추출합니다.
red_mask = cv2.inRange(img2, (0, 0, 200), (50, 50, 255))

# 검은색 배경을 흰색으로 변경합니다.
result_img[red_mask == 0] = [255, 255, 255]

# 전처리된 이미지를 파일로 저장합니다.
cv2.imwrite('result_image.png', result_img)
#%%
img2 = cv2.imread('./result_image.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
#%%
print(img1.shape)
plt.imshow(img1)
plt.show()

# %% 사진 사이즈 동일하게
img1 = cv2.resize(img1,(640,640))
img2 = cv2.resize(img2,(640,640))
# %%
print(img1.shape)
plt.imshow(img1)
plt.show()
# %%
print(img2.shape)
plt.imshow(img2)
plt.show()

# %%
blended = cv2.addWeighted(src1=img1, alpha=0.5,src2=img2, beta=0.5,gamma=0) # 반반적용
# %%
plt.imshow(blended) # 두 이미지가 겹쳐 하나의 이미지
plt.show()
# %%
blended2 = cv2.addWeighted(src1=img1, alpha=0.8,src2=img2, beta=0.1,gamma=0) 
plt.imshow(blended2) 
plt.show()
# %%
url = 'https://cdn.pixabay.com/photo/2018/09/26/09/07/education-3704026_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('letters.jpg')
# %%
img = cv2.imread('./letters.jpg',0)
img.shape
# %%
plt.imshow(img,cmap='gray')
plt.show()
# %%
ret, thresh1 = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img,128,255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img,128,255,cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img,128,255,cv2.THRESH_TOZERO_INV)
# %%
titles=['Origin','Bin','Bin_inv','trunc','tozero','tozero_inv']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
# %%
plt.figure(figsize=(14,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    
plt.show()
# %%
url = 'https://cdn.pixabay.com/photo/2014/12/02/22/05/snowflakes-554635_960_720.jpg'
response = requests.get(url)
pic = Image.open(BytesIO(response.content))
pic.save('snow.jpg')
# %%
img2 = cv2.imread('./snow.jpg',0)
img2.shape
# %%
ret, thresh1 = cv2.threshold(img2,128,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img2,128,255,cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img2,128,255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img2,128,255,cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img2,128,255,cv2.THRESH_TOZERO_INV)
# %%
titles=['Origin','Bin','Bin_inv','trunc','tozero','tozero_inv']
images = [img2,thresh1,thresh2,thresh3,thresh4,thresh5]
# %%
plt.figure(figsize=(14,7))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    
plt.show()
# %%
