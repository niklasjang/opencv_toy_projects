# opencv_toy_projects

Python OpenCV toy projects

## Obecjt-Tracking

![Object-Tracking-1](/assets/images/object-tracking-1.jpg)  

Feature Matching BruteForce
    1. Grayscale two images
    2. Create ORB object
    3. DetectAndCompute each image
    4. Set BFMatcher
    5. bf.match
    6. sort matches
    7. drawMatches
    8. imshow
1. 주어지는 영상에서 추적할 대상을 ROI를 사용해서 독립적인 이미지로 저장합니다.
2. 두 영상을 Grayscale로 변환합니다.
3. ORB object를 만들고 두 영상을 DetectAndCompute합니다.
4. 각각의 영상에서 찾아진 특징점들을 Brute-Force 알고리즘을 사용해서 매칭시킵니다.
5. 매칭된 쌍들을 거리에 따라 정렬을 합니다. 정렬은 작은 Window를 구성할 좌표를 결정하는데 사용됩니다.
6. 매칭된 결과를 모두 포함하는 가장 작은 원을 출력합니다.

## Cal-Palate Detecting

![car-palate-detecting-1](/assets/images/car-palate-detecting-1.jpg)  

1. 주어지는 영상을 h,s,v 성분으로 분리합니다.
2. 번호판의 숫자를 잘 추출할 수 있도록 전처리과정(morphology, gaussian filter, threshold)을 진행합니다.
3. 전처리가 완료된 영상에서 contour를 추출합니다.
4. 입력으로 주어질 영상의 번호판이 과도하게 틀어져있지 않다는 점에서 착안하여 입력 영상에서 번호판의 contour일 가능성이 높은 것을 추립니다.
5. 일반적인 번호판의 비율과 비슷하게 찾을 수 있는 컨투어들의 집합을 추출합니다.
6. 번호판을 찾았으면 Perspective 변환을 진행한 뒤 출력합니다.
7. 5번의 과정에서 실패한다면 번호판을 찾을 수 없다는 결과를 보여줍니다.

