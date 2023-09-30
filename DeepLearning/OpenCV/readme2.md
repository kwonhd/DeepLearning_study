# 컬러 매핑(Color Mapping)
    - 주로 그레이 스케일(Grayscale), 트루 컬러(True Color, RGB)이미지를 많이 활용
    - 다양한 색 공간(ex, HSV, YCrCB 등)이 존재하고 이들을 변환할 수 있음
    - 컬러 영상 처리에서 HSV와 HSL은 같은 색 공간을 이용하여 색상 구분에 용이하고, YCrCb와 YUV는 휘도 성분 구분에 용이
    - cv2.cvtColor() 활용

    - 색 공간의 종류 (참고)
        - RGB
            - 컬러 표현을 빛의 3원색인 빨강(Red), 초록(Green), 파랑(Blue)으로 서로 다른 비율을 통해 색 표현
            - 가산 혼합(additive mixture): 빛을 섞을 수록 밝아짐
            - 모니터, 텔레비전, 빔 프로젝터와 같은 디스플레이 장비들에서 기본 컬러 공간으로 사용
        
        - CMYK
            - 청록색(Cyan), 자홍색(Magenta), 노랑색(Yellow), 검은색(Black)을 기본으로 하여 주로 컬러 프린터나 인쇄시에 사용
            - 감산 혼합(subtractive mixture): 섞을 수록 어두워지는 방식
            - RGB 컬러 공간과 보색 관계

  <img src="https://blog.kakaocdn.net/dn/cCzazP/btqt3Ii1kR9/sn0gSXL7OuSpFkdjEqqkFK/img.png" width="500">

        - YUV
            - Y축은 밝기 성분을 U,V 두축을 이용하여 색상을 표현
            - U축은 파란색에서 밝기 성분을 뺀 값, V축은 빨간색에서 밝기 성분을 뺀 값
            - 아날로그 컬러신호 변환에 주로 사용. (U = B - Y) , (V = R - Y)
        - YCbCr
            - Digital TV에서 사용하는 색공간
            - YPbPr이라는 아날로그 신호의 색공간을 디지털화한 것
            - YPbPr은 아날로그 컴포넌트 비디오에서 사용



# RGB Color Space
    - 디지털 컬러 영상을 획득할 때 사용
    - 보편적으로 사용되고 있지만 '컬러 영상 처리'에서는 주로 사용되지 않음
  <br>
  
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/RGB_Cube_Show_lowgamma_cutout_b.png/600px-RGB_Cube_Show_lowgamma_cutout_b.png" width="400">

  <sub>[이미지 출처] https://en.wikipedia.org/wiki/RGB_color_space</sub>


# HSV Color Space
    - 색상(Hue), 채도(Saturation), 명도(Value)로 색을 표현
    - 색상은 흔히 빨간색, 노란색 등과 같은 색의 종류
    - 채도는 색의 순도
    - 예들 들어, 파란색에서
        - 채도가 높으면 맑고 선한 파란색
        - 채도가 낮으면 탁한 파란색
    - 명도는 빛의 세기,
        - 명도가 높으면 밝고, 낮으면 어둡게 느껴진다.
    - OpenCV에서 BGR2HSV 색 공간 변환할 경우,
        - H : 0 ~ 179 사이의 정수로 표현
            - 색상 값은 0° ~ 360° 로 표현하지만 uchar 자료형은 256이상의 정수를 표현할 수 없기 때문에 OpenCV에서는 각도를 2로 나눈 값을 H로 저장
        - S : 0 ~ 255 사이의 정수로 표현
        - V : 0 ~ 255 사이의 정수로 표현
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/HSV_cone.jpg/400px-HSV_cone.jpg">

# HSL Color Space
    - 색상(Hue), 채도(Saturation), 밝기(Lightness)로 색을 표현하는 방식
    - HSV와 동일하지만 밝기 요소의 차이
    - HSV와 더불어 사람이 실제로 color를 인지하는 방식과 유사

<img src="https://rgbtohex.page/imgs/hsl-cylinder.png">

# YCrCb Color Space
    - Y 성분은 밝기 또는 휘도(luminance), Cr, Cb 성분은 색상 또는 색차(chrominance)를 나타냄
    - Cr, Cb는 오직 색상 정보만 가지고 있음. 밝기 정보 X
    - 영상을 GrayScale 정보와 색상 정보로 분리하여 처리할 때 유용
    - Y, Cr, Cb : 0 ~ 255 사이의 정수로 표현

  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/YCbCr-CbCr_Scaled_Y50.png/600px-YCbCr-CbCr_Scaled_Y50.png" width="300">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YCbCr.GIF">

# GrayScale Color Space
    - 영상의 밝기 정보를 256단계 (0 ~ 255) 로 구분하여 표현
    - 가장 밝은 흰색 : 255, 가장 어두운 검은색 : 0


<img src="https://miro.medium.com/max/1400/1*euc4RxnNo78LFEGrb-QZ7w.jpeg" width="600">

# 스토그램 (Histogram)
    - 이미지의 밝기의 분포를 그래프로 표현한 방식
    - 이미지의 전체의 밝기 분포와 채도(밝고 어두움)을 알 수 있음

<img src="https://opencv-python.readthedocs.io/en/latest/_images/image013.jpg">

    - 용어설명
        BINS
            - 히스토그램 그래프의 X축의 간격
            - 위 그림의 경우에는 0 ~ 255를 표현하였기 때문에 BINS값은 256이 된다.
            - BINS값이 16이면 0 ~ 15, 16 ~ 31..., 240 ~ 255와 같이 X축이 16개로 표현
            - OpenCV에서는 BINS를 histSize 라고 표현

        DIMS
            - 이미지에서 조사하고자하는 값을 의미
            - 빛의 강도를 조사할 것인지, RGB값을 조사할 것인지를 결정

        RANGE
            - 측정하고자하는 값의 범위
        
        - cv2.calcHist()

|파라미터|설명|
|-------|-------|
|image|분석대상 이미지(uint8 or float32 type). Array형태|
|channels|분석 채널(X축의 대상), 이미지가 graysacle이면 [0], color 이미지이면 [0],[0,1] 형태(1 : Blue, 2: Green, 3: Red)
|mask|이미지의 분석영역. None이면 전체 영역|
|histSize|BINS 값.[256]|
|ranges|range값.[0,256]|

# 히스토그램 평탄화
    - 이미지의 히스토그램이 특정영역에 너무 집중되어 있으면 contrast가 낮아 좋은 이미지라고 할 수 없음
    - 전체 영역에 골고루 분포가 되어 있을 때 좋은 이미지, 아래 히스토그램을 보면 좌측 처럼 특정 영역에 집중되어 있는 분포를 오른쪽 처럼 골고루 분포하도록 하는 작업을 Histogram Equalization 이라고 함

    - (참고) 이론적인 방법
        - 이미지의 각 픽셀의 cumulative distribution function(cdf)값을 구하고 Histogram Equalization 공식에 대입하여 0 ~ 255 사이의 값으로 변환

        - 이렇게 새롭게 구해진 값으로 이미지를 표현하면 균일화된 이미지를 얻을 수 있음

  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Histogrammeinebnung.png/600px-Histogrammeinebnung.png">

  <sub>[이미지 출처] https://en.wikipedia.org/wiki/Histogram_equalization</sub>

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - 지금까지의 처리는 이미지의 전체적인 부분에 균일화를 적용
    - 일반적인 이미지는 밝은 부분과 어두운 부분이 섞여 있기 때문에 전체에 적용하는 것은 그렇게 유용하지 않음
    - 이 문제를 해결하기 위해서 adaptive histogram equalization을 적용하게 됨
        - 즉, 이미지를 작은 tile형태로 나누어 그 tile안에서 Equalization을 적용하는 방식
        - 작은 영역이다 보니 작은 노이즈(극단적으로 어둡거나, 밝은 영역)가 있으면 이것이 반영이 되어 원하는 결과를 얻을 수 없게 됨
        - 이 문제를 피하기 위해서 contrast limit라는 값을 적용하여 이 값을 넘어가는 경우는 그 영역은 다른 영역에 균일하게 배분하여 적용

# 2D Histogram : cv2.calcHist()

    - 지금까지 Histogram은 1차원으로 grayscale 이미지의 pixel의 강도, 즉 빛의 세기를 분석한 결과
    - 2D Histogram은 Color 이미지의 Hue(색상) & Saturation(채도)을 동시에 분석하는 방법
    - 색상과 채도를 분석하기 때문에 HSV Format으로 변환해야 함

|파라미터|설명|
|-------|-------|
|image|HSV로 변환된 이미지|
|channel|0-> Hue, 1-> Saturation|
|bins|	[180,256] 첫번째는 Hue, 두번째는 Saturation|
|range|	[0,180,0,256] Hue(0~180), Saturation(0,256)|

# 이미지 처리
    - 필요에 따라 적절한 처리
    - resize(), flip(), getAffineTransform(), warpAffine() 등 다양한 메서드 존재

    # Resize
    cv2.resize()
        - 사이즈가 변하면 pixel사이의 값을 결정을 해야함
        - 보간법(Interpolation method)
        - 사이즈를 줄일 때 : cv2.INTER_AREA
        - 사이즈를 크게 할 때 : cv2.INTER_CUBIC , cv2.INTER_LINEAR

|파라미터|설명|
|-------|-------|
|img|Image|
|dsize|	Manual Size, 가로, 세로 형태의 tuple(e.g., (100,200))|
|fx|	가로 사이즈의 배수, 2배로 크게하려면 2. 반으로 줄이려면 0.5|
|fy|	세로 사이즈의 배수|
|interpolation|	보간법|

# Translation : cv2.warpAffine()
    - 이미지의 위치를 변경
|파라미터|설명|
|-------|-------|
|src|Image|
|M|변환 행렬|
|dsize (tuple)|output image size(e.g., (width=columns, height=rows)|


# Rotate : cv2.getRotationMatrix2D()
    - 물체를 평면상의 한 점을 중심으로 𝜃 만큼 회전하는 변환
    - 양의 각도는 시계반대방향으로 회전

|파라미터|설명|
|-------|-------|
|center|	이미지의 중심 좌표|
|angle|	회전 각도|
|scale|	scale factor|

# Flip : cv2.flip()
    - 대칭 변환
        - 좌우 대칭 (좌우 반전)
        - 상하 대칭 (상하 반전)
    - 입력 영상과 출력 영상의 픽셀이 1:1 매칭이므로 보간법이 필요 없음

|파라미터|설명|
|-----|-----|
|src|입력 영상|
|flipCode|대칭 방법을 결정하는 flag 인자, 양수이면 좌우 대칭, 0이면 상하 대칭, 음수이면 상하, 좌우 대칭을 모두 실행|

# Affine Transformation : cv2.getAffineTransform()

    - 선의 평행선은 유지되면서 이미지를 변환하는 작업
    - 이동, 확대, Scale, 반전까지 포함된 변환
    - Affine 변환을 위해서는 3개의 Match가 되는 점이 있으면 변환행렬을 구할 수 있음

#Perspective Transformation
    - Perspective(원근법) 변환
    - 직선의 성질만 유지, 선의 평행성은 유지가 되지 않는 변환
    - 기차길은 서로 평행하지만 원근변환을 거치면 평행성은 유지 되지 못하고 하나의 점에서 만나는 것 처럼 보임 (반대의 변환도 가능)
    - 4개의 Point의 Input값과 이동할 output Point 가 필요
    - cv2.getPerspectiveTransform()가 필요하며, cv2.warpPerspective() 함수에 변환행렬값을 적용하여 최종 결과 이미지를 얻을 수 있음
    - 좌표점은 (왼쪽 위)→(오른쪽 위)→(오른쪽 아래)→(왼쪽 아래)

#이미지 ROI
    - 이미지 작업시에는 특정 pixel단위 보다는 특정 영역단위로 작업을 하게 되는데 이것을 Region of Image(ROI)라고 함
    - ROI 설정은 Numpy의 indexing을 사용, 특정 영역을 copy 할 수도 있음