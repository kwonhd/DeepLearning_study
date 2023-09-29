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