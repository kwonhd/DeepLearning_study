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

# 이미지 더하기
    - cv2.add()
    - cv2.addWeighted()
    - Numpy 더하기 연산
    - cv2.add() : Saturation 연산

        - Saturation 연산은 한계값을 정하고 그 값을 벗어나는 경우는 모두 특정 값으로 계산하는 방식
        - 이미지에서는 0이하는 모두 0, 255이상은 모두 255로 표현
    - Numpy : modulo 연산
        - a와 b는 n으로 나눈 나머지 값이 같다라는 의미
        - 이미지에서는 연산의 결과가 256보다 큰 경우는 256으로 나눈 나머지 값으로 결정

# 비트 연산
    - AND, OR, NOT, XOR 연산
        - bitwise_and : 둘다 0이 아닌 경우만 값을 통과
        - bitwise_or : 둘중 하나가 0이 아니면 값을 통과
        - bitwise_not : 해당 값에 대해 부정값을 통과
        - bitwise_xor : 두 요소의 논리적 배타값 통과

# 이미지 블렌딩(Image Blending) : cv2.addWeighted()
    - 두 이미지를 blending 할 수 있음
    - blending 하려는 두 이미지의 사이즈가 같아야함
    - [Simple Formula]
        - g(x)=(1−α)f0(x)+αf1(x) 
        - β=1−α 
        -α,β  의 값을 통해 어떤 이미지를 더 강하게 드러내고, 어떤 이미지를 더 약하게 드러낼지 결정
        - γ  추가 가능 (optional)

# 이미지 이진화 (Image Thesholding)

# 기본 임계 처리 : cv2.threshold()

    - 이진화 : 영상을 흑/백으로 분류하여 처리하는 것
        - 기준이 되는 임계값을 어떻게 결정할 것인지가 중요한 문제
        - 임계값보다 크면 백, 작으면 흑이 되는데, 기본 임계처리는 사용자가 고정된 임계값을 결정하고 그 결과를 보여주는 단순한 형태
        기본적으로 이미지의 segmenting의 가장 간단한 방법

|파라미터|설명|
|-------|-------|
|src|	input image로 single-channel 이미지.(grayscale 이미지)|
|thresh|	임계값|
|maxval|	임계값을 넘었을 때 적용할 value|
|type|	thresholding type|


    - thresholding type
        - cv2.THRESH_BINARY
            - src(x, y) > thresh 일 때, maxval
            - 그 외, 0
        - cv2.THRESH_BINARY_INV
            - src(x, y) > thresh 일 때, 0
            - 그 외, maxval
        - cv2.THRESH_TRUNC
            - src(x, y) > thresh 일 때, thresh
            - 그 외, src(x, y)
        - cv2.THRESH_TOZERO
            - src(x, y) > thresh 일 때, src(x, y)
            - 그 외, 0
        - cv2.THRESH_TOZERO_INV
            - src(x, y) > thresh 일 때, 0
            - 그 외, src(x, y)

# 적응 임계처리 : cv2.adaptiveThreshold()

    - 이전 단계에서는 임계값을 이미지 전체에 적용하여 처리하기 때문에
    하나의 이미지에 음영이 다르면 일부 영역이 모두 흰색 또는 검정색으로 보여지게 됨
    - 이런 문제를 해결하기 위해서 이미지의 작은 영역별로 thresholding

|파라미터|설명|
|-------|-------|
|src|	grayscale image|
|maxValue|	임계값|
|adaptiveMethod|	thresholding value를 결정하는 계산 방법|
|thresholdType|	threshold type|
|blockSize	|thresholding을 적용할 영역 사이즈|
|C|	평균이나 가중평균에서 차감할 값|
    - Adaptive Method|
        - cv2.ADAPTIVE_THRESH_MEAN_C : 주변영역의 평균값으로 결정
        - cv2.ADAPTIVE_THRESH_GAUSSIAN_C : 주변영역의 가우시안 값으로 결정

# Otsu의 이진화
    - Otsu의 이진화(Otsu’s Binarization)란 bimodal image에서 임계값을 자동으로 계산하는 것
    - 임계값을 결정하는 가장 일반적인 방법은 trial and error 방식
    - bimodal image (히스토그램으로 분석하면 2개의 peak가 있는 이미지)의 경우는 히스토그램에서 임계값을 어느정도 정확히 계산 가능
    - cv2.threshold() 함수의 flag에 추가로 cv2.THRESH_OTSU 를 적용. 이때 임계값은 0으로 전달

# 이미지 필터링(Image Filtering) : cv2.filter2D()

    - 이미지도 음성 신호처럼 주파수로 표현할 수 있음

    - 일반적으로 고주파는 밝기의 변화가 많은 곳, 즉 경계선 영역에서 나타나며, 일반적인 배경은 저주파로 나타냄
        - 이것을 바탕으로 고주파를 제거하면 Blur처리가 되며, 저주파를 제거하면 대상의 영역을 확인 가능
    - Low-pass filter(LPF)와 High-pass filter(HPF)를 이용하여,
    LPF를 적용하면 노이즈제거나 blur처리를 할 수 있으며, HPF를 적용하면 경계선을 찾을 수 있음
    - 일반적으로 많이 사용되는 필터

            |1 1 1 1 1|
            |1 1 1 1 1|
ex) K = 1/25|1 1 1 1 1|
            |1 1 1 1 1|
            |1 1 1 1 1|


# 이미지 샤프닝(Image Sharpening)
    - 출력화소에서 이웃 화소끼리 차이를 크게 해서 날카로운 느낌이 나게 만드는 것

    - 영상의 세세한 부분을 강조할 수 있으며, 경계 부분에서 명암대비가 증가되는 효과

    - 사프닝 커널

        - 커널 원소들의 값 차이가 커지도록 구성
        - 커널 원소 전체합이 1이 되어야 입력영상 밝기가 손실 없이 출력영상 밝기로 유지

# 이미지 블러링(Image Blurring)
    - low-pass filter를 이미지에 적용하여 얻을 수 있음
    - 고주파영역을 제거함으로써 노이즈를 제거하거나 경계선을 흐리게 할 수 있음
    - OpenCV에서 제공하는 blurring 방법
        - Averaging
        - Gaussian Filtering
        - Median Filtering
        - Bilateral Filtering

- Averaging
    - Box형태의 kernel을 이미지에 적용한 후 평균값을 box의 중심점에 적용하는 형태
    - cv2.blur() 또는 cv2.boxFilter()
    - cv2.blur()
        - Parameters
            - src : Chennel수는 상관없으나,
            depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
            - ksize : kernel 사이즈(ex; (3,3))

- ex) K = 1/9|1 1 1|
             |1 1 1|
             |1 1 1|

- 이미지의 Data Type

|데이터 타입|설명|
|-------|-------|
|CV_8U|	8-bit unsigned integer: uchar ( 0..255 )|
|CV_8S|	8-bit signed integer: schar ( -128..127 )|
|CV_16U|	16-bit unsigned integer: ushort ( 0..65535 )|
|CV_16S|	16-bit signed integer: short ( -32768..32767 )|
|CV_32S|	32-bit signed integer: int ( -2147483648..2147483647 )|
|CV_32F|	32-bit floating-point number: float ( -FLT_MAX..FLT_MAX, INF, NAN )|
|CV_64F|	64-bit floating-point number: double ( -DBL_MAX..DBL_MAX, INF, NAN )|

- 일반적으로 Data Type과 채널수가 같이 표현이 되어 CV_8UC1 과 같이 표현
(8bit unsiged integer이면서 채널이 1개)

# Gaussian Filtering : cv2.GaussianBlur()
    - box filter는 동일한 값으로 구성된 kernel을 사용하지만, Gaussian Filter는 Gaussian함수를 이용한 Kernel을 적용
        - kernel 행렬의 값을 Gaussian 함수를 통해서 수학적으로 생성하여 적용
    kernel의 사이즈는 양수이면서 홀수로 지정을 해야 됨
    - 이미지의 Gaussian Noise (전체적으로 밀도가 동일한 노이즈, 백색노이즈)를 제거하는 데 가장 효과적

|파라미터|설명|
|-------|-------|
|img|	Chennel수는 상관없으나, depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F|
|ksize|	(width, height) 형태의 kernel size, width와 height는 서로 다를 수 있지만, 양수의 홀수로 지정해야 함|
|sigmaX|	Gaussian kernel standard deviation in X direction|

# Median Filtering :cv2.medianBlur()
    - kernel window와 pixel의 값들을 정렬한 후에 중간값을 선택하여 적용
    - salt-and-pepper noise 제거에 가장 효과적

|파라미터|설명|
|-------|-------|
|src|	1, 3, 4 channel image, depth가 CV_8U, CV_16U, or CV_32F 이면 ksize는 3또는5, CV_8U이면 더 큰 ksize가능|
|ksize|	1보다 큰 홀수|

# Bilateral Filtering : cv2.bilateralFilter()

    - 위 3가지 Blur 방식은 경계선까지 Blur 처리가 되어, 경계선이 흐려지게 됨
    - Bilateral Filtering(양방향 필터)은 경계선을 유지하면서 Gaussian Blur처리를 해주는 방법


|파라미터|설명|
|-------|-------|
|src|	8-bit, 1 or 3 Channel image|
|d|	filtering시 고려할 주변 pixel 지름|
|sigmaColor|	Color를 고려할 공간. 숫자가 크면 멀리 있는 색도 고려|
|sigmaSpace|	숫자가 크면 멀리 있는 pixel도 고려|


# 형태학적 변환(Morphological Transformations)
    -이미지를 Segmentation하여 단순화, 제거, 보정을 통해서 형태를 파악하는 목적으로 사용
    -일반적으로 binary나 grayscale image에 사용
    -사용하는 방법으로는 Dilation(팽창), Erosion(침식), 그리고 2개를 조합한 Opening과 Closing이 있음
    -여기에는 2가지 Input값이 있는데, 하나는 원본 이미지이고 또 다른 하나는 structuring element
    -structuring element
        - 원본 이미지에 적용되는 kernel
        - 중심을 원점으로 사용할 수도 있고, 원점을 변경할 수도 있음
        - 일반적으로 꽉찬 사각형, 타원형, 십자형을 많이 사용

# Erosion : cv2.erode()

- 각 Pixel에 structuring element를 적용하여 하나라도 0이 있으면 대상 pixel을 제거하는 방법
- 작은 object를 제거하는 효과


|파라미터|설명|
|-------|-------|
|src|	the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F|
|kernel|	structuring element. cv2.getStructuringElemet() 함수로 만들 수 있음|
|anchor|	structuring element의 중심. default (-1,-1)로 중심점|
|iterations|	erosion 적용 반복 횟수|

    -아래 그림은 대상 이미지에 십자형 structuring element를 적용한 결과

<img src="https://opencv-python.readthedocs.io/en/latest/_images/image01.png">

<sub>[이미지 출처] http://www.kocw.net/home/search/kemView.do?kemId=1127905&ar=relateCourse</sub>

# Dilation : cv2.dilation()

    - Erosion과 반대 작용
    - 대상을 확장한 후 작은 구멍을 채우는 방법
    - Erosion과 마찬가지로 각 pixel에 structuring element를 적용
    - 대상 pixel에 대해서 OR 연산을 수행
    - 즉, 겹치는 부분이 하나라도 있으면 이미지를 확장

  <img src="https://opencv-python.readthedocs.io/en/latest/_images/image03.png">

|파라미터|설명|
|--------|----|
|`src`|the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F|
|`kernel`|structuring element. cv2.getStructuringElemet() 함수로 만들 수 있음|
|`anchor`|structuring element의 중심. default (-1,-1)로 중심점|
|`iterations`|dilation 적용 반복 횟수|


### Opening & Closing

`cv2.morphologyEx()`

- Opening과 Closing은 Erosion과 Dilation의 조합 결과
- 차이는 어느 것을 먼저 적용을 하는 차이
- `Opening` : Erosion적용 후 Dilation 적용. 작은 Object나 돌기 제거에 적합
- `Closing` : Dilation적용 후 Erosion 적용. 전체적인 윤곽 파악에 적합

<img src="https://opencv-python.readthedocs.io/en/latest/_images/image05.png">

|파라미터|설명|
|--------|----|
|`src`|원본 이미지. 채널수는 상관 없으나, depth는 다음 중 하나여야 함 `CV_8U`, `CV_16U`, `CV_16S`, `CV_32F`, `CV_64F`
|`op`|연산 방법|
|`MORPH_OPEN`|열기 동작|
|`MORPH_CLOSE`|닫기 동작|
|`MORPH_GRADIENT`|a morphological gradient. Dilation과 erosion의 차이|
|`MORPH_TOPHAT`|“top hat”, Opeining과 원본 이미지의 차이|
|`MORPH_BLACKHAT`|“black hat”. Closing과 원본 이미지의 차이|
|`kernel`|structuring element. `cv2.getStructuringElemet()` 함수로 만들 수 있음|
|`anchor`|structuring element의 중심. default (-1,-1)로 중심점|
|`iterations`|erosion과 dilation 적용 횟수|
|`borderType`|픽셀 외삽법 `borderInterpolate` 참고(https://ko.wikipedia.org/wiki/%EB%B3%B4%EC%99%B8%EB%B2%95)|
|`borderValue`|테두리 값|

### Morphological Gradient

- dilation과 erosion의 차이

### Top Hat

- 입력 이미지와 opening 이미지와의 차이

### Black Hat
    - 입력 이미지와 closing 이미지와의 차이

### Structuring Element
    - 사각형 모양의 structuring element는 numpy 를 통해 만들 수 있음
    - 원이나 타원 모양이 필요한 경우, `cv2.getStructuringElement()` 이용
    - Parameters
        - `shape` :
    Element의 모양.
        - `MORPH_RET` : 사각형 모양
        - `MORPH_ELLIPSE` : 타원형 모양
        - `MORPH_CROSS` : 십자 모양
        - `ksize` : structuring element 사이즈


# 이미지 기울기(Image Gradients)

- Gradient(기울기)는 영상의 edge 및 그 방향을 찾을 때 활용됨

- 이미지 (x, y)에서의 벡터값(밝기와 밝기의 변화하는 방향)을 구해서 해당 pixel이 edge에 얼마나 가까운지, 그 방향이 어디인지 알 수 있음

### Soble & Scharr Filter

`cv2.Sobel()`

- Gaussian smoothing과 미분을 이용
- 노이즈가 있는 이미지에 적용하면 좋음
- X축과 Y축을 미분하는 방법으로 경계값을 계산

|파라미터|설명|
|--------|----|
|`src`|input image|
|`ddepth`|output image의 depth, -1이면 input image와 동일|
|`dx`|x축 미분 차수|
|`dy`|y축 미분 차수|
|`ksize`|kernel size(ksize x ksize)|

- `cv2.Scharr()`: `cv2.Sobel()`과 동일하지만 `ksize`가 soble의 3x3보다 정확하게 적용됨

### Laplacian 함수

`cv2.Laplacian()`

- 이미지의 가로와 세로에 대한 Gradient를 2차 미분한 값
- Sobel filter에 미분의 정도가 더해진 것과 비슷함 
- (dx와 dy가 2인 경우) blob(주위의 pixel과 확연한 pixel차이를 나타내는 덩어리)검출에 많이 사용됨

|파라미터|설명|
|--------|----|
|`src`|source image|
|`ddepth`|output image의 depth|

### Canny Edge Detection

`cv2.Canny()`

- 가장 유명한 Edge Detection 방법
- `Noise Reduction`  
  - 이미지의 Noise를 제거
  - 이때 5x5의 Gaussian filter를 이용
- `Edge Gradient Detection`
  - 이미지에서 Gradient의 방향과 강도를 확인
  - 경계값에서는 주변과 색이 다르기 때문에 미분값이 급속도로 변하게 됨
  - 이를 통해 경계값 후보군을 선별
- `Non-maximum Suppression`
  - 이미지의 pixel을 Full scan하여 Edge가 아닌 pixel은 제거
- `Hysteresis Thresholding`
  - 이제 지금까지 Edge로 판단된 pixel이 진짜 edge인지 판별하는 작업 진행
  - max val과 minVal(임계값)을 설정하여 maxVal 이상은 강한 Edge, min과 max사이는 약한 edge로 설정
  - 이제 약한 edge가 진짜 edge인지 확인하기 위해서 강한 edge와 연결이 되어 있으면 edge로 판단하고, 그러지 않으면 제거

|파라미터|설명|
|--------|----|
|`image`|8-bit input image|
|`threshold1`|Hysteresis Thredsholding 작업에서의 min 값|
|`threshold2`|Hysteresis Thredsholding 작업에서의 max 값|