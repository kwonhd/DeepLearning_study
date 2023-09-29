# Open CV
    - 영상처리
        - 저수준 영상처리
            - 영상 획득
            - 영상 향상
            - 영상 복원
            - 변환 처리
            - 영상 압축
        - 고수준 영상처리(컴퓨터 비전)
            - 영상 분할
            - 영상 표현
            - 영상 인식

    - 컴퓨터 비전 처리 단계
        - 전처리 단계
            - 주로 영상처리 기술 사용
            - 다양한 특징 추출 : 에지(edge), 선분, 영역, SIFT(Scale-Invariant Feature Transform) 등
        - 고수준 처리
            - 특징정보를 사용하여 영상을 해석, 분류, 상황묘사 등 정보 생성
    - 이미지와 색공간
        - color : 3차원, RGB, 0~255
        - gray scale : 2차원, 0~255

    - Open CV : 실시간 이미지 프로세싱에 중점
        - 이미지 읽기/쓰기(Numpy)
        - 이미지 읽기(PIL) : 코랩, 주피터는 PIL, matplotlib 적합

            - 이미지 출력 : cv2.imshow

            - 이미지 읽기 
                - cv2.IMREAD_COLOR: 이미지 파일을 Color로 읽어들이고, 투명한 부분은 무시되며, Default 값
                - cv2.IMREAD_GRAYSCALE: 이미지를 Grayscale로 읽음. 실제 이미지 처리시 중간단계로 많이 사용
                - cv2.IMREAD_UNCHANGED: 이미지 파일을 alpha channel (투명도)까지 포함하여 읽어 들임
                - (주의) cv2.imread()는 잘못된 경로로 읽어도 NoneType으로 들어갈 뿐, 오류를 발생하지 않음
            - 이미지 쓰기
                - cv2.imwrite()

- 이미지 선 긋기

| 파라미터 | 설명 |
|---------|------|
| img     | 그림을 그릴 이미지 파일 |
|star|시작 좌표|
|end|종료 좌표|
|color|BGR형태의 Color(e,g,(255,0)->Blue)|
|thickness(int)|선의 두께(pixel)|

- 사각형 그리기

| 파라미터 | 설명 |
|---------|------|
| img     | 그림을 그릴 이미지 파일 |
|star|시작 좌표|
|end|종료 좌표|
|color|BGR형태의 Color(e,g,(255,0,0)->Blue)|
|thickness(int)|선의 두께(pixel)|

- 원 그리기

| 파라미터 | 설명 |
|---------|------|
| img     | 그림을 그릴 이미지 파일 |
|centenr|원의 중심 좌표(x,y)|
|radian|반지름|
|color|BGR형태의 Color|
|thickness(int)|선의 두께, -1이면 원 안쪽을 채움|
|lineType|선의 형태, cv2.line()함수의 인수와 동일|
|shift|좌표에 대한 비트 시프트 연산|

- 타원 그리기

| 파라미터 | 설명 |
|---------|------|
| img     | 그림을 그릴 이미지 파일 |
|centenr|타원의 중심 좌표(x,y)|
|axes|중심에서 가장 큰 거리와 작은 거리|
|angle|타원의 기울기 각|
|startAngle|타원의 시작 각도|
|endAngle|타원의 끝나는 각도|
|color|색상|
|thickness|선의 두께, -1이면 안쪽 채움|
|lineType|선의 형태|
|shift|좌표에 대한 비트 시프트|

- 다각형 그리기

| 파라미터 | 설명 |
|---------|------|
| img     | 그림을 그릴 이미지 파일 |
|pts(array)|연결할 꼭지점 좌표|
|isClosed|닫힌 도형 여부(True,False)|
|color|색상|
|thickness|선 두께|

- 텍스트 그리기

| 파라미터 | 설명 |
|---------|------|
| img     | 그림을 그릴 이미지 파일 |
|text|표시할 문자열|
|org|문자열이 표시될 위치, 문자열의 bottom-left corner 점|
|fontFace|폰트타입. cv2.font_xxx|
|fontScale|폰트크기|
|color|폰트색상|
|thickness|글자의 굵기|
|lineType|글자 선의 형태|
|bottomLeftOrign|영상의 원점 좌표 설정(True:최하단, False:최상단)|

- 문자열 폰트 옵션

|옵션|값|설명|
|-------|-------|-------|
|cv2.FONT_HERSHEY_SIMPLEX	    |0|	중간 크기 산세리프 폰트|
|cv2.FONT_HERSHEY_PLAIN	        |1|	작은 크기 산세리프 폰트|
|cv2.FONT_HERSHEY_DUPLEX	    |2|	2줄 산세리프 폰트|
|cv2.FONT_HERSHEY_COMPLEX	    |3|	중간 크기 세리프 폰트|
|cv2.FONT_HERSHEY_TRIPLEX	    |4|	3줄 세리프 폰트|
|cv2.FONT_HERSHEY_COMPLEX_SMALL	|5|	COMPLEX 보다 작은 크기|
|cv2.FONT_HERSHEY_SCRIPT_SIMPLEX|6|	필기체 스타일 폰트|
|cv2.FONT_HERSHEY_SCRIPT_COMPLEX|7|복잡한 필기체 스타일|
|cv2.FONT_ITALIC	            |16|	이탤릭체를 위한 플래그|