# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:10:43 2019

@author: maker
"""
import cv2
import numpy as np
import math  

def getKey(item):
    return item[0]

def getMax(x, y):
    if( x> y ) :
        return x
    else :
        return y

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  


def addAllContours(contoursList, contours) :
    #===== maybe_chars에 들어있는 index를 가지는 countour들을 하나의 conntour로 만듦
    check =False
    all_cnt = 0
    for char_index in contoursList :
        #print(contours[char_index].shape)
        if(check == False):
            all_cnt = contours[char_index]
            check = True
        else :
            all_cnt = np.append(all_cnt,contours[char_index], axis = 0)
    return all_cnt
    
def getMinAreaRectangleBox(all_cnt):
    #===== all_cnt를 감싸는 Box을 찾아서 네 모서리와 함께 그린다. 
    rect = cv2.minAreaRect(all_cnt) # Rotated Rectangle 영역을 감싸는 가장 작은 사각형을 찾는다. 
    box= cv2.boxPoints(rect) #이 사각형을 감싸는 box  points를 찾는다.
    box = np.int0(box)       #소수점을 제거해줘야 그려진다.

    #===== box의 좌표를 항상 같은 포멧으로 결정되게 함.
    # box= cv2.boxPoints(rect)로 얻은 box 배열에는 (0)~(3)의 좌표값들이 랜덤한 순서로 저장된다. 
    # 이 좌표값들을 항상 아래와 같은 순서로 저장되도록 한다.
    # (0) ------선분03------- (3)     파 ------------------ 검
    #  |                       |      |                     |
    # 선분01                 선분23   |                     |
    #  |                       |      |                     |
    # (1) -------선분12------ (2)     초 ------------------ 빨
    
    # x좌표를 기준으로 정렬하여 (0),(1) 과 (2),(3)의 두 묶음으로 나눈다.
    box = sorted (box,key=lambda x: (x[0], x[1]))
    points = box.copy()
    points = np.int0(points) #소수점 절삭
    
    # 원점과 (0), 원점과 (1) 사이의 거리를 계산하여 더 가까운 점이 왼쪽위의 점이 된다.
    point_distance_0 = calculateDistance(0,0, box[0][0], box[0][1])
    point_distance_1 = calculateDistance(0,0, box[1][0], box[1][1])
    if point_distance_0 > point_distance_1 :
        points[0][0],points[1][0] = points[1][0], points[0][0]
        points[0][1],points[1][1] = points[1][1], points[0][1]
    
    # 원점과 (2), 원점과 (3) 사이의 거리를 계산하여 더 가까운 점이 왼쪽위의 점이 된다.
    point_distance_2 = calculateDistance(0,0, box[2][0], box[2][1])
    point_distance_3 = calculateDistance(0,0, box[3][0], box[3][1])
    if point_distance_2 < point_distance_3 : 
        points[2][0],points[3][0] = points[3][0], points[2][0]
        points[2][1],points[3][1] = points[3][1], points[2][1]
    points = np.int0(points) #소수점 절삭
    return points

def calculateWidthHeightRatio(contours_list, contours):
    
    all_cnt = addAllContours(contours_list, contours)
    
    points = getMinAreaRectangleBox(all_cnt)
    
    width = getMax( calculateDistance(points[0][0], points[0][1], points[3][0], points[3][1]) ,
                                calculateDistance(points[1][0], points[1][1], points[2][0], points[2][1]) )
    height = getMax( calculateDistance(points[0][0], points[0][1], points[1][0], points[1][1]) ,
                                calculateDistance(points[3][0], points[3][1], points[2][0], points[2][1]) )
    ratio = width/ height
    return width, height, ratio

def drawPointsCircle(points, shrink):
    cv2.circle(shrink, (points[0][0], points[0][1]), 10, (255,0,0), -1) #파란색 원
    cv2.circle(shrink, (points[1][0], points[1][1]), 10, (0,255,0), -1) #초록색 원
    cv2.circle(shrink, (points[2][0], points[2][1]), 10, (0,0,255), -1) #빨간색 원
    cv2.circle(shrink, (points[3][0], points[3][1]), 10, (0,0,0), -1)   #검은색 원
    
def getPointsNewPoints(maybe_chars, contours):
    
    all_cnt = addAllContours(maybe_chars, contours)
    points = getMinAreaRectangleBox(all_cnt)
    points = np.float32(points)
    
    width, height, ratio = calculateWidthHeightRatio(maybe_chars, contours)
    
    width = int(width)
    height = int(height)
    newPoints = np.int0([[0,0], [0,height],[width,height], [width,0]]) #output이 될 window의 좌표
    newPoints = np.float32(newPoints)
    
    return width, height, points, newPoints

def getContours(image):
    #===== value 추출
    h,s,value = cv2.split(image)
    #===== Image 전처리해서 contours 찾기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 0. Erosion :각 Pixel에 structuring element를 적용하여 하나라도 0이 있으면 대상 pixel을 제거하는 방법입니다. 
        # 0. Diltion :각 Pixel에 structuring element를 적용하여 하나라도 1이 있으면 영역을 확장합니다. 
        # 0. Opeing : Erosion적용 후 Dilation 적용. 작은 Object나 돌기 제거에 적합
        # 0. Closing : Dilation적용 후 Erosion 적용. 전체적인 윤곽 파악에 적합
        # 1. MORPH_OPEN - an opening operation
        # 2. MORPH_CLOSE - a closing operation
        # 3. MORPH_TOPHAT - “top hat”. Opeining과 원본 이미지의 차이
        # 4. MORPH_BLACKHAT - “black hat”. Closing과 원본 이미지의 차이
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, topHat) # topHat을 더하고
    subtract = cv2.subtract(add, blackHat) # blackHat을 뻄
    blur = cv2.GaussianBlur(subtract, (5, 5), 0) # Gaussian Blur
        # thresholding : 이미지의 작은 영역별로 thresholding. 임계값 전후로 모든 값이 통일 되는 것을 방지
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        # 0. Contours :  동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선
        # 1. 정확도를 높히기 위해서 Binary Image를 사용합니다. threshold나 canny edge를 선처리로 수행합니다.
        # 2. contours를 찾는 것은 검은색 배경에서 하얀색 대상을 찾는 것
        # 3. [Warning!] the function cv2.findContours() has been changed to return only the contours and the hierarchy .
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def trimContours(contours):
    #===== 필요한 Contours들만 남겨서 글자의 contour일 가능성이 있는 것들을 uplefts에 저장
    dirst_scope = [30,200]                                                                     #적당한 길이의 cnt 찾
    area_scope = [500,2500]                                                                    #contour의 넓이로 필터링 : 길쭉하거나 큰 contours는 지우기
    hull_area_criteria = 350                                                                   #hull의 넓이로 필터링 : 너무 큰 hull은 지우기
    uplefts = []                                                                               #여기에 포함된 index 번째의 contours는 char가 될 가능성이 있는 것
    for i in range(len(contours)):
        if (dirst_scope[0] < len(contours[i]) and len(contours[i]) < dirst_scope[1] ) :        #적당한 길이의 cnt
            x,y,w,h = cv2.boundingRect(contours[i])                                            #contour를 감싸는 x, y, 사각형 가로길이, 사각형 세로 길이    
            if(w*0.7 < h and h < w*5.0) :                                                      #적당한 가로 세로 길이
                if(area_scope[0] < w*h and w*h < area_scope[1] ):                              #넓이로 필터링 : 길쭉하거나 큰 contours는 지우기
                    hull = cv2.convexHull(contours[i])
                    hull_area = cv2.contourArea(hull)    
                    #min_rect = cv2.minAreaRect(contours[i])
                    #min_box = cv2.boxPoints(min_rect)
                    #min_box = np.int0(min_box)
                    if(hull_area > hull_area_criteria) :                                       #hull의 넓이로 필터링 : 너무 큰 hull은 지우기
                        #contour_img = cv2.drawContours(shrink, contours, i, (0,0,255), 1)
                        #contour_img = cv2.rectangle(shrink, (x, y), (x+w, y+h), (0,255,0), 1)
                        #cv2.drawContours(shrink, [hull], 0,(0,255,0), 3)
                        #cv2.drawContours(shrink, [min_box], 0,(255,0,0), 3)
                        uplefts.append((x,y,w,h,i))
    return uplefts

def getMaybeChars(trimedContours, contours):
    #===== uplefts 중에서 글자의 contour일 가능성이 있는 것들을 추려서  maybe_chars에 저장.
    trimedContours = sorted(trimedContours,  key = getKey)
    max_detect_count = 0                                                                       #max_detect_height에서 얼마나 많은 사각형을 찾았는지
    maybe_chars = []                                                                           #max_detect_height에서 찾은 contour의 index가 들어있음
    temp_chars = []                                                                            #maybe_chars에 넣기 이전의 버퍼
    temp_output_ratio_criteria = [2.5, 6.5]
    for height in range(600,0, -5):                                                        #위에서부터 내려오면서
        temp_detect = 0                                                                        #현재의 height에서 몇 개의 사각형이 겹쳤는지 임시 저장
        temp_chars.clear()
        for x,y,w,h,detected_i in trimedContours :                                                    #모든 uplefts 점에 대해
            if(y<= height and height <= y+h):                                                  #몇 개의 사각형이 겹치는지 판단
                temp_detect += 1
                temp_chars.append(detected_i) 
        if( max_detect_count < temp_detect and temp_detect >= 4) :                              # 4개 이상 겹쳤고 지금까지 이렇게 많이 겹친게 없었으면
            #print("temp_detect", temp_detect)
            #print("temp_chars", temp_chars)
            temp_output_width, temp_output_height, temp_output_ratio = calculateWidthHeightRatio(temp_chars, contours)
            if ( temp_output_ratio_criteria[0] <= temp_output_ratio  and 
                 temp_output_ratio <=temp_output_ratio_criteria[1]   and 
                 max_detect_count < temp_detect):
                #print("temp_output_ratio", temp_output_ratio)
                #contour_img = cv2.drawContours(shrink, contours, detected_i, (0,0,255), 10)
                #==== 다 더한 컨투어들로 사각형 구하고 이들의 가로세로비율이 2.5~6.2 사이이면 maybe_cahrs 에 넣기                     
                max_detect_count = temp_detect
                #print("temp_chars", temp_chars)
                maybe_chars = temp_chars.copy()
    return maybe_chars

def getResult(shrink, maybe_chars, contours):
    if(len(maybe_chars) != 0):
        #글자 찾은 경우
        output_width,output_heigth, points, newPoints = getPointsNewPoints(maybe_chars, contours)
        # ==== warpPerspective 수행해서 output image을 뽑음
        M = cv2.getPerspectiveTransform(points, newPoints)
        output = cv2.warpPerspective(shrink, M, (output_width,output_heigth))
        result = output.copy()
    else :
        #글자 못찾은 경우 : 오류출력
        no_plate_white = np.zeros((50,400,3), np.uint8)
        no_plate_white [:] = (255, 255, 255)
        result = no_plate_white.copy()
        cv2.putText(result, "Can't find License plate", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return result

def main():
    #===== HYPER PARAMETERS
    WIDTH = 800
    HEIGHT = 600
    #path = "D:\\OpenCV\\images\\cars\\"
    while(True) :
       
        imgpath = input("Enter a file name: ") # E.g. .//S01.jpg
        
        #===== READ IMAGE and Resize
        original = cv2.imread(imgpath, 1)
        shrink = cv2.resize(original, (WIDTH,HEIGHT), interpolation=cv2.INTER_AREA)
        
        contours = getContours(shrink)
        
        uplefts = trimContours(contours)
        
        maybe_chars = getMaybeChars(uplefts, contours)
        
        result = getResult(shrink, maybe_chars, contours)
        
        #===== maybe_chars에 있는 contour들을 붉은색 글씨로 표시
        for detected_i in (maybe_chars):
            shrink = cv2.drawContours(shrink, contours, detected_i, (0,0,255), 3) #수평선과 겹친 사각형을 빨간색으로 테투리를 그림
            #hull = cv2.convexHull(all_cnt)
            #contour_img = cv2.drawContours(shrink, [hull], 0,(0,255,), 1)

        cv2.imshow("Orignial", shrink)
        cv2.imshow("Car plate", result)

        k = cv2.waitKey(0)
        
        if k == 27: # esc key
            break
        elif k == ord('r') : # 's' key
            cv2.destroyAllWindows()
            continue
        elif k == 13 : # 's' key
            cv2.destroyAllWindows()
            continue 
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()