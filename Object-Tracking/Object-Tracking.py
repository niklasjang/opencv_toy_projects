# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:12:55 2019

@author: Admin

Week13
"""
import cv2
import numpy as np
 
def getFeatureMatchingCircleCoordinate(kp2, matches):
    # Initialize lists
    xpos = []
    ypos = []
    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x2,y2) = kp2[img2_idx].pt
    
        # Append to each list
        xpos.append(x2)
        ypos.append(y2)
        xpos = sorted(xpos, key = lambda x:x)
        ypos = sorted(ypos, key = lambda x:x)
        
        if( xpos[-1] - xpos[0] > ypos[-1] - ypos[0]) :
            radian = int((xpos[-1] - xpos[0]) / 2)
        else :
            radian = int((ypos[-1] - ypos[0]) / 2)
        
    return radian

def main():
    # Create tracket object
    tracker = cv2.TrackerKCF_create()
    
    filepath = "D:\\OpenCV\\videos\\"
    
    # Select video 
    #videoname = "1st_school_tour_headquarter.avi"
    #videoname = "school.avi"
    videoname = "red_bus.avi"
    #videoname = "school_central_park.avi"
    
    filename = filepath + videoname
    video = cv2.VideoCapture(filename)
    
    # Check video is opened or not.
    if not video.isOpened():
        print("CANNOT FIND VIDEO")
        return
    # Get vedio's Frame Per Second
    fps = video.get(cv2.CAP_PROP_FPS) # obj로부터 frame rate – fps 읽어오기
    
    # Read video. This have to be done here to make user select ROI box
    ret, frame = video.read()
    if not ret:
        print("CANNOT READ VIDEO")
        return

    # Put red color instruction on every video frame
    instruction = "Draw a rectangle over the object you want to detect and then press ENTER"
    cv2.putText(frame, instruction, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    '''
    Draw a box on the first frame of the video
    box is the object that we want to detect
    box consists of 2 coordinates and 2 values that are width and height
    '''
    box = cv2.selectROI(frame, False)
    
    # Close ROI select window
    cv2.destroyWindow('ROI selector')
    
    # Reset the tracker to track the selected area 
    ret = tracker.init(frame, box)
    box_copied = box
    #box_detected = box

    # Get detecting object's position in frame from selectROI box components
    x = int(box[0])
    y = int(box[1])
    w = int(box[2])
    h = int(box[3])
    
    # Save the pixels in the selected area to detecting_obj
    detecting_obj = frame[y : y+h, x : x+w]
    selected_obj =detecting_obj
    
    '''
    Feature Matching BruteForce
    1. Grayscale two images
    2. Create ORB object
    3. DetectAndCompute each image
    4. Set BFMatcher
    5. bf.match
    6. sort matches
    7. drawMatches
    8. imshow
    '''
    selected_obj_gray = cv2.cvtColor(selected_obj, cv2.COLOR_BGR2GRAY)    # 1
    #detecting_obj_gray = cv2.cvtColor(detecting_obj, cv2.COLOR_BGR2GRAY)  # 1
    orb = cv2.ORB_create()                                                # 2
    kp1, des1 = orb.detectAndCompute(selected_obj_gray, None)             # 3
    #kp2, des2 = orb.detectAndCompute(detecting_obj_gray, None)            # 3
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)               # 4
    #matches = bf.match(des1, des2)                                        # 5
    #matches = sorted(matches, key = lambda x:x.distance)                  # 6
    #matching_result = cv2.drawMatches(selected_obj_gray, kp1, detecting_obj_gray, kp2, matches[:30], None, flags=2) # 7
    #cv2.imshow("Matching result", matching_result)                        # 8
    
    # Open a new window to display detecting_obj
    # cv2.imshow('selected object',selected_obj)
    
    # Define closure windows
    closure_frame = np.zeros((64,384,3), np.uint8)
    cv2.putText(closure_frame, "Tracking Failed", (60,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("CANNOT READ VIDEO")
            break
        # Update tracker
        # retval, boundingBox	=	cv.Tracker.update(	image	)
        # boundingBox : The bounding box that represent the new target location, if true was returned, not modified otherwise
        
        # The more slow_rate is lower than 1.0 the  more video become slower.
        slow_rate = 1.0
        
        ret, box = tracker.update(frame)
        # If object found : Tracking success
        if ret:
            print("Tracking success")
            nx = int(box[0]) # new x
            ny = int(box[1]) # new y
            nw = int(box[2]) # new w
            nh = int(box[3]) # new h
            
            if( nx < 0 ) : nx = 0
            if( ny < 0 ) : ny = 0
            detecting_obj = frame[ny : ny+nh, nx : nx+nw]
            p1 = (nx, ny)
            p2 = (nx + nw, ny + nh)
            print("first {}, {}, {}, {}".format(nx,ny,nw,nh))
            #print("first {}, {}".format(nx, ny))
            #print("second {}, {}".format(nx + nw, ny + nh))
            detecting_obj_gray = cv2.cvtColor(detecting_obj, cv2.COLOR_BGR2GRAY)  # 1
            kp2, des2 = orb.detectAndCompute(detecting_obj_gray, None)            # 3
            matches = bf.match(des1, des2)                                        # 5
            radian = getFeatureMatchingCircleCoordinate(kp2, matches)
            
            center_xpos = int(nx + (nw/2))
            center_ypos = int(ny + (nh/2))
            circle_center = center_xpos, center_ypos
            #matches = sorted(matches, key = lambda x:x.distance)                  # 6
            #print(matches[0].imgIdx )
            
            
            # cv2.drawMatches(img1, keypoints1, img2, kp2, matches1to2[, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]]) 
            matching_result = cv2.drawMatches(selected_obj_gray, kp1, 
                                              detecting_obj_gray, kp2, 
                                              matches1to2 = matches[:30], 
                                              outImg = None,
                                              matchColor = ((255, 0, 0)), 
                                              flags = 2) # 7
            cv2.imshow("Matching result", matching_result)                        # 8
            
            # cv2.rectangle(img, start, end, color, thickness)
            #cv2.rectangle(frame, p1, p2, (255,0,0), 1) 
            cv2.circle(frame, circle_center, radian , (0,255,255), 3)
        # If object not found
        else :
            print("Tracking fail")
            
            # get the grayscale of detecting_obj, detect feature points using ORB detector
            grayscale = cv2.cvtColor(detecting_obj, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(grayscale, None)
            # extract the feature points from the current frame
            grayscale2= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(grayscale2, None)
            
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)
            
            # when detected
            if(len(matches) > 0):
                print("mathces found")
                print("matches[0]: ", matches[0])
                if(matches[0] is not int):
                    cv2.imshow('CLOSURE', closure_frame)
                    cv2.waitKey(0)
                    break
                x=kp2[matches[0].queryIdx].pt[0] - kp1[matches[0].queryIdx].pt[0]
                y=kp2[matches[0].trainIdx].pt[1] - kp1[matches[0].trainIdx].pt[1]
                box=(x,y, box_copied[2], box_copied[3])
                p1 = (x, y)
                p2 = (x + w, y + h)
                # re-detected object
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                matching_result = cv2.drawMatches(detecting_obj, kp1, frame, kp2, matches, None, flags=2)
                cv2.imshow('Match Result',matching_result)
                # re-initialize the tracker to track the successful result
                tracker.init(frame, box)
                
            else:
                print("mathces fail")
                cv2.putText(closure_frame, "Tracking Failed", (60,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if cv2.waitKey((int)(fps/24 * slow_rate)) == 27:
                    break
            
        cv2.putText(frame, "Press ESC to Exit", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # Live Tracking
        cv2.imshow("Live Tracking", frame)
        # Exit if ESC pressed
        
        #k = cv2.waitKey((int)(fps/24 * slow_rate))
        k = cv2.waitKey(10)
        if k == 27 : break
    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__' :
    main()