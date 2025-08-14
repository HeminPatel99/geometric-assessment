import cv2
import csv
import os
import random
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage

if __name__ == "__main__":
  # Specify the path to the folder containing your images
  dir_ = os.path.dirname(__file__)
  dom1_extension = ['NF.jpg', 'NF.jpeg', 'NF.png']
  dom2_extension = ['FF.jpg', 'FF.jpeg', 'FF.png']

  folder_path1 = os.path.join(dir_,'GT6K/')#change name to the folder name
  # List all files in the folder
  file_list = os.listdir(folder_path1)
  # Sort the files based on their filenames
  sorted_files = sorted(file_list)
  # Filter out only the image files

  dom1_files = [file for file in sorted_files if any(file.endswith(ext) for ext in dom1_extension)]
  dom2_files = [file for file in sorted_files if any(file.endswith(ext) for ext in dom2_extension)]

  print(dom1_files)

  # create the necessary folders
  f1 = os.path.join(dir_, 'dom1/')
  f2 = os.path.join(dir_, 'dom2/')
  fp1 = 's1_preprocess/'
  fp2 = 's2_contrast/'
  fp3 = 's3_segmentation/'
  fp4 = 's4_smooth/'
  fp5 = 's5_annotate_images/'
  fp6 = 's6_data/'
  f_list = [f1,f2]
  fp_list = [fp1,fp2,fp3,fp4,fp5,fp6]
  save_dom1 = []
  save_dom2 = []
  # Check if the folder exists, and if not, create it
  for f in f_list:
    if not os.path.exists(f):
        os.makedirs(f)
  c = 0
  for f in f_list:
    c += 1
    for fp in fp_list:
      folder = os.path.join(f,fp)
      if c == 1:
        save_dom1.append(folder)
      else:
        save_dom2.append(folder)
      if not os.path.exists(folder):
        os.makedirs(folder)

  # Step 1: Image Preprocessing
  # img = cv2.imread('your_image.jpg', 0)  # Load as grayscale (cool)

  dom1 = []
  dom2 = []
  temp_dom1 = []
  temp_dom2 = []
  for d1 in dom1_files:
    img_path = os.path.join(folder_path1, d1)
    temp = cv2.imread(img_path)
    dom1.append(temp)
    #print(np.array(dom1).shape)
  for d2 in dom2_files:
    img_path = os.path.join(folder_path1, d2)
    temp = cv2.imread(img_path)
    dom2.append(temp)
    #print(np.array(dom2).shape)
  x11 = []
  x12 = []
  y11 = []
  y12 = []
  dom1_gray_list = []
  for d1 in dom1:
    dom1_gray = cv2.cvtColor(d1, cv2.COLOR_BGR2GRAY)
    dom1_gray_list.append(dom1_gray)
    gray = np.array(dom1_gray)
    height, width = gray.shape
    fs = 1
    check = 0
    for col in range(width):
      for row in range(height):
        if dom1_gray[row][col] > 10:
          if (fs == 1):
            y11.append(col)
            fs = 2
            break
          elif (fs == 2):
            break
        if (fs == 2) & (row == (height-1)):
          y12.append(col)
          check = 1
          fs = 1#reset
          break
      if check == 1:
        check = 0#reset
        break

    for row in range(height):
      for col in range(width):
        if dom1_gray[row][col] > 10:
          if (fs == 1):
            x11.append(row)
            fs = 2
            break
          elif (fs == 2):
            break
        if (fs == 2) & (col == (width-1)):
          x12.append(row)
          check = 1
          fs = 1#reset
          break
      if check == 1:
        check = 0#reset
        break

  #resize dom1
  for dom1_g in dom1_gray_list:
    tp1 = []
    for x in range(math.floor(min(x11)), math.ceil(max(x12))):#rows
      tp2 = []
      for y in range(math.floor(min(y11)), math.ceil(max(y12))):#cols
        tp2.append(dom1_g[x][y])
      tp1.append(tp2)
    temp_dom1.append(tp1)

  x21 = []
  x22 = []
  y21 = []
  y22 = []
  dom2_gray_list = []
  for d2 in dom2:
    dom2_gray = cv2.cvtColor(d2, cv2.COLOR_BGR2GRAY)
    dom2_gray_list.append(dom2_gray)
    gray = np.array(dom2_gray)
    height, width = gray.shape
    fs = 1
    check = 0
    for col in range(width):
      for row in range(height):
        if dom2_gray[row][col] > 10:
          if (fs == 1):
            y21.append(col)
            fs = 2
            break
          elif (fs == 2):
            break
        if (fs == 2) & (row == (height-1)):
          y22.append(col)
          check = 1
          fs = 1#reset
          break
      if check == 1:
        check = 0#reset
        break

    for row in range(height):
      for col in range(width):
        if dom2_gray[row][col] > 10:
          if (fs == 1):
            x21.append(row)
            fs = 2
            break
          elif (fs == 2):
            break
        if (fs == 2) & (col == (width-1)):
          x22.append(row)
          check = 1
          fs = 1#reset
          break
      if check == 1:
        check = 0#reset
        break

  #resize dom2
  for dom2_g in dom2_gray_list:
    tp1 = []
    for x in range(math.floor(min(x21)), math.ceil(max(x22))):#rows
      tp2 = []
      for y in range(math.floor(min(y21)), math.ceil(max(y22))):#cols
        tp2.append(dom2_g[x][y])
      tp1.append(tp2)
    temp_dom2.append(tp1)

  temp_dom1= np.array(temp_dom1)
  temp_dom2= np.array(temp_dom2)

  new_dom1 = []
  new_dom2 = []

  ts1 = ((max(y12)-min(y11))*4,(max(x12)-min(x11))*4) #target sizes
  ts2 = ((max(y22)-min(y21))*2, (max(x22)-min(x21))*2) #target sizes

  new_dom1= np.array(temp_dom1)
  new_dom2= np.array(temp_dom2)

  l1 = len(new_dom1)
  l2 = len(new_dom2)

  print(new_dom1.shape)
  print(new_dom2.shape)
  print("l1: ", l1)
  print("l2 : ", l2)

  #save the image
  count = 0
  for d1 in new_dom1:
      count += 1
      img_name = 'preprocessed_NF_' + str(count) + '.jpg'
      cv2.imwrite(os.path.join(save_dom1[0], img_name), d1 )
  count = 0
  for d2 in new_dom2:
      count += 1
      img_name = 'preprocessed_FF_' + str(count) + '.jpg'
      cv2.imwrite(os.path.join(save_dom2[0], img_name), d2 )

  print("Preprocessing Complete")

  # FOR DOM 2 ROIs

  temp_center = (291-min(y21), 214-min(x21))
  gh = [71, 73, 36] #manually find and record here
  bh = [50, 24, 36] #manually find and record here
  temp_green_hex = [(temp_center[0],temp_center[1]-gh[0]),(temp_center[0]+gh[1],temp_center[1]-gh[2]),(temp_center[0]+gh[1],temp_center[1]+gh[2]),(temp_center[0],temp_center[1]+gh[0]),(temp_center[0]-gh[1],temp_center[1]+gh[2]),(temp_center[0]-gh[1],temp_center[1]-gh[2])]
  for i in range(1,4):
      hex = [(temp_center[0]-bh[1]*i,temp_center[1]-bh[2]*i),(temp_center[0]+bh[1]*i,temp_center[1]-bh[2]*i),(temp_center[0]+bh[0]*i,temp_center[1]),(temp_center[0]+bh[1]*i,temp_center[1]+bh[2]*i),(temp_center[0]-bh[1]*i,temp_center[1]+bh[2]*i),(temp_center[0]-bh[0]*i,temp_center[1])]
      if i == 1:
          temp_blue_hex = hex
      elif i == 2:
          temp_red_hex = hex
      else:
          temp_outter_hex = hex
  blue_hex = []
  green_hex = []
  red_hex = []
  outter_hex = []
  for i in range (len(temp_blue_hex)):
      blue_hex.append((temp_blue_hex[i][0]*2,temp_blue_hex[i][1]*2))
  for i in range (len(temp_green_hex)):
      green_hex.append((temp_green_hex[i][0]*2,temp_green_hex[i][1]*2))
  for i in range (len(temp_red_hex)):
      red_hex.append((temp_red_hex[i][0]*2,temp_red_hex[i][1]*2))
  for i in range (len(temp_outter_hex)):
      outter_hex.append((temp_outter_hex[i][0]*2,temp_outter_hex[i][1]*2))
  center = (temp_center[0]*2,temp_center[1]*2)

  print("Regions of Interest DOM 2")
  print("center: ", center)
  print(blue_hex)
  print(green_hex)
  print(red_hex)
  print(outter_hex)

  #Step 2 : Contrast and Sharpen
  #Constrast Enhancement for better image segmentation
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #cliplimit = 1 (original); 2 or 3 are good; anymore can cause oversaturation
  # Apply CLAHE
  c1_img = []
  c2_img = []
  for img in new_dom1:
      c1_img.append(clahe.apply(img))
  for img in new_dom2:
      c2_img.append(clahe.apply(img))

  c1_img = np.array(c1_img)
  c2_img = np.array(c2_img)
  #save the image
  for i in range(l1):
      img_name = 'contrast_NF_' + str(i+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom1[1], img_name), c1_img[i] )

  for j in range(l2):
      img_name = 'contrast_FF_' + str(j+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom2[1], img_name), c2_img[j] )

  print("Contrast Enhancement Complete")

  # Sharpen Image
  #DOM1
  sharp1_img = []
  for i in range(l1):
      # Apply Gaussian smoothing
      blurred_image1 = cv2.GaussianBlur(c1_img[i], (5, 5), 0)

      # Apply the Laplacian operator
      laplacian1 = cv2.Laplacian(blurred_image1, cv2.CV_64F)

      # Calculate the sharpened image by adding the Laplacian to the original image
      sharp1_img.append(cv2.addWeighted(c1_img[i], 1, laplacian1, -5.5, 0, dtype=cv2.CV_8U))  # You can adjust the alpha and beta values (!!!!)

      img_name = 'sharp_NF_' + str(i+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom1[1], img_name), sharp1_img[i] )

  #DOM2
  sharp2_img = []
  for j in range(l2):
      # Apply Gaussian smoothing
      blurred_image2 = cv2.GaussianBlur(c2_img[j], (5, 5), 0)

      # Apply the Laplacian operator
      laplacian2 = cv2.Laplacian(blurred_image2, cv2.CV_64F)

      # Calculate the sharpened image by adding the Laplacian to the original image
      sharp2_img.append(cv2.addWeighted(c2_img[j], 1, laplacian2, -2.5, 0, dtype=cv2.CV_8U))  # You can adjust the alpha and beta values(!!!!)

      img_name = 'sharp_FF_' + str(j+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom2[1], img_name), sharp2_img[j])

  print("Sharpening Complete")

  #for DOM 2 AutoMation
  print(new_dom1.shape)
  #TO AUTOMATE
  avg1 = []
  avg2 = []
  for a in sharp1_img:
    c1 = 0
    math1 = 0
    for b in a:
      for c in b:
        math1 += c
        c1 += 1
    avg1.append(math1/c1)

  for a in sharp2_img:
    c2 = 0
    math2 = 0
    for b in a:
      for c in b:
        math2 += c
        c2 += 1
    avg2.append(math2/c2)

  print("dom1: ", avg1)
  print("dom2: ", avg2)

  # Step 3: Pattern Detection (Segmentation)
  # Thresholding
  binary_image1 = []
  binary_image2 = []
  counter = 0
  for image in sharp1_img:
      # Denoising
      denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
      sm = ndimage.median_filter(denoised_image, size=10)
      # Frequency domain analysis
      f = np.fft.fft2(sm)
      fshift = np.fft.fftshift(f)
      magnitude_spectrum = 20 * np.log(np.abs(fshift))

      # Thresholding to isolate periodic patterns
      # y = avg1[counter]*(-0.29) + 170.75
      # threshold = y
      threshold = 190
      fshift[np.where(magnitude_spectrum < threshold)] = 0

      # Inverse FFT
      f_ishift = np.fft.ifftshift(fshift)
      img_back = np.fft.ifft2(f_ishift)
      img_back = np.abs(img_back)

      # Thresholding to create binary image of artifacts
      #ret, segmented = cv2.threshold(np.uint8(img_back), 80, 255, cv2.THRESH_BINARY)
      segmented = cv2.adaptiveThreshold(np.uint8(img_back),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,33,5)
      # y = 1.58*avg1[counter] + 63.68
      # #b_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,5)
      # _,b_img = cv2.threshold(img, y, 255, cv2.THRESH_BINARY)#(!!!!)()
      binary_image1.append(segmented)
      # counter += 1
  counter = 0
  for img in sharp2_img:
      y = avg2[counter]*0.96 + 13 # found by finding the best line of fit .52
      b_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,43,7)
      #_,b_img = cv2.threshold(img, y, 255, cv2.THRESH_BINARY)#(!!!!) (100,120,75,20)
      binary_image2.append(b_img)
      counter += 1

  # Resize dom images
  ic = 0
  new_binary_image1 = []
  for img in binary_image1:
    img = cv2.resize(img, ts1, interpolation=cv2.INTER_AREA)
    new_binary_image1.append(img)
    ic += 1
  ic = 0
  new_binary_image2 = []
  for img in binary_image2:
    img = cv2.resize(img, ts2, interpolation=cv2.INTER_AREA)
    new_binary_image2.append(img)
    ic += 1

  # print(binary_image)
  #print(len(binary_image1))
  for i in range(l1):
      img_name = 'binary_NF_' + str(i+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom1[2], img_name), new_binary_image1[i])
  for j in range(l2):
      img_name = 'binary_FF_' + str(j+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom2[2], img_name), new_binary_image2[j])

  print("Resizing and Segmentation Complete")

  #Step 4: Smoothing

  smooth1 = []
  smooth2 = []
  i = 0
  for img in new_binary_image1:
      smoothed = ndimage.median_filter(img, size=10)
      smooth1.append(smoothed)
      img_name = 'smooth_NF_' + str(i+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom1[3], img_name), smooth1[i] )
      i += 1
  j = 0
  for img in new_binary_image2:
      smoothed = ndimage.median_filter(img, size=10)
      smooth2.append(smoothed)
      img_name = 'smooth_FF_' + str(j+1) + '.jpg'
      cv2.imwrite(os.path.join(save_dom2[3], img_name), smooth2[j] )
      j += 1

  print("Smoothing Complete")

  # Step 5: Shape Recognition
  contours_list1 = []
  contours_list2 = []
  for i in range(l1):
    contours, _ = cv2.findContours(smooth1[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_list1.append(contours)
  for j in range(l2):
    contours, _ = cv2.findContours(smooth2[j], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_list2.append(contours)

  shapes_img_list1 = []
  shapes_img_list2 = []

  for i in range(2):
    if i == 0:
      img = contours_list1 # filter or not
    else:
      img = contours_list2 # filter or not
    for contours in img:
      a = 0
      b = 0
      c = 0
      d = 0
      e = 0
      stemp = [] # temp var
      for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.03 * cv2.arcLength(contour, True) # edit: lower the more vertices
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Determine shape based on the number of vertices
        num_vertices = len(approx)
        shape_type = "Unknown"
        if num_vertices == 3:
          shape_type = "Tri" + str(a)
          a += 1
        elif num_vertices == 4:
          shape_type = "Rect" + str(b)
          b += 1
        elif num_vertices == 5:
          shape_type = "Pent" + str(e)
          e += 1
        elif num_vertices == 6:
          shape_type = "Hex" + str(c)
          c += 1
        elif num_vertices > 6:
          shape_type = "Cir" + str(d)
          d += 1
        if shape_type != "Unknown":
          stemp.append((shape_type, approx))
      if i == 0:
        shapes_img_list1.append(stemp)
      else:
        shapes_img_list2.append(stemp)

  print(len(shapes_img_list2))

  # Step 4: Distance Measurement
  # Calculate distances between midpoints of shapes and the midpoint of the image

  distance_from_mid_list1 = []
  #distance_from_mid_list2 = []
  centroids_img_list1 = []
  centroids_img_list2 = []
  area1 = []
  area2 = []
  new_shapes1 = []
  new_shapes2 = []

  for i in range(2):
      if i == 0:
          sm = smooth1
          shapes_list = shapes_img_list1
      else:
          sm = smooth2
          shapes_list = shapes_img_list2
      counter = 0
      for smooth_img in sm: # go through each image from that dom
          image_height, image_width = smooth_img.shape
          image_midpoint1 = (image_width // 2, image_height // 2) # get midpoint of this image
          img_dist_list = []
          img_centroids_list = []
          min_area = []
          temp_ns = []
          for shape_type, approx in shapes_list[counter]:# go through the list of contours for that image
              moments = cv2.moments(approx)
              if i == 0:
                  lower = 120
                  upper = 20000
              else:
                  lower = 120
                  upper = 20000
              if (lower < moments["m00"] < upper):
                  temp_ns.append((shape_type,approx))
                  shape_midpoint = (
                      int(moments["m10"] / moments["m00"]), # m00 is the area so we can use this to compare if the hexagons are of equal size
                      int(moments["m01"] / moments["m00"])
                  )
                  img_centroids_list.append((shape_midpoint,approx))
                  if (shape_midpoint[0] == 551) & (shape_midpoint[1] == 428):
                      print(approx)
                  # Calculate the distance between shape midpoint and image midpoint
                  dist = np.sqrt((shape_midpoint[0] - image_midpoint1[0])**2 + (shape_midpoint[1] - image_midpoint1[1])**2)
                  min_area.append((shape_type, moments["m00"]))
                  img_dist_list.append((shape_type, dist))
          counter += 1
          if i == 0:
              distance_from_mid_list1.append(img_dist_list)
              centroids_img_list1.append(img_centroids_list)
              area1.append(min_area)
              new_shapes1.append(temp_ns)
          else:
              #distance_from_mid_list2.append(img_dist_list)
              centroids_img_list2.append(img_centroids_list)
              area2.append(min_area)
              new_shapes2.append(temp_ns)

  print(len(centroids_img_list2[0]))

  image_height, image_width = np.array(smooth2[0]).shape
  image_midpoint2 = (image_width // 2, image_height // 2) # get midpoint of this image

  #DOM 2- finding the centriods in the ROIs

  roi = 25
  roi_list = []
  checkc = 0
  b_hex = []
  g_hex = []
  r_hex = []
  o_hex = []
  c_img = []
  for img in centroids_img_list2:
      # check Center
      r_list = []
      shortest_d = 25
      cent = image_midpoint2
      for centroid, approx in img:
          distance = np.sqrt((centroid[0] - center[0])**2 + (centroid[1] - center[1])**2)
          if distance <= shortest_d:
              checkc = 1
              shortest_d = distance
              t_a = approx
              cent = centroid
      c_img.append(cent)
      name = "center"
      if checkc == 1:
          r_list.append((name, cent, t_a))
          checkc = 0

      h_counter = 0
      tb = []
      for centroid_blue in blue_hex:
          v = 0
          centr = cent
          for centroid, approx in img:
              distance = np.sqrt((centroid[0] - centroid_blue[0])**2 + (centroid[1] - centroid_blue[1])**2)
              if distance <= roi:
                  v = 1
                  centr = centroid
                  name = "blue" + str(h_counter)
                  r_list.append((name, centroid, approx))
                  break
          tb.append((v,centr))
          h_counter += 1
      b_hex.append(tb)
      h_counter = 0
      tg = []
      for centroid_green in green_hex:
          v = 0
          centr = cent
          for centroid, approx in img:
              distance = np.sqrt((centroid[0] - centroid_green[0])**2 + (centroid[1] - centroid_green[1])**2)
              if distance <= roi:
                  v = 1
                  centr = centroid
                  name = "green" + str(h_counter)
                  r_list.append((name, centroid, approx))
                  break
          tg.append((v,centr))
          h_counter += 1
      g_hex.append(tg)
      h_counter = 0
      tr = []
      for centroid_red in red_hex:
          v = 0
          centr = cent
          for centroid, approx in img:
              distance = np.sqrt((centroid[0] - centroid_red[0])**2 + (centroid[1] - centroid_red[1])**2)
              if distance <= roi:
                  v = 1
                  centr = centroid
                  name = "red" + str(h_counter)
                  r_list.append((name, centroid, approx))
                  break
          tr.append((v,centr))
          h_counter += 1
      r_hex.append(tr)
      h_counter = 0
      to = []
      for centroid_outter in outter_hex:
          v = 0
          centr = cent
          for centroid, approx in img:
              distance = np.sqrt((centroid[0] - centroid_outter[0])**2 + (centroid[1] - centroid_outter[1])**2)
              if distance <= roi:
                  v = 1
                  centr = centroid
                  name = "outter" + str(h_counter)
                  r_list.append((name, centroid, approx))
                  break
          to.append((v,centr))
          h_counter += 1
      o_hex.append(to)
      roi_list.append(r_list) # name and centroid

  distance_from_mid_list2 = []

  for img in roi_list:
      temp_dd = []
      for shape_type, cent, approx in img:
          dd= np.sqrt((cent[0] - image_midpoint2[0])**2 + (cent[1] - image_midpoint2[1])**2)
          temp_dd.append((shape_type, dd))
      distance_from_mid_list2.append(temp_dd)

  #step 7 Annotate the binary image
  ann_imgs1 = []

  for img in smooth1:
    ann = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ann_imgs1.append(ann)

  shapes_list = new_shapes1

  ann_imgs = ann_imgs1
  dist_list = distance_from_mid_list1
  counter1 = 0
  for img in ann_imgs:
    annotated_image = img.copy()

    for shape_type, approx in shapes_list[counter1]:
      # Draw the shape
      cv2.drawContours(annotated_image, [approx], -1, (0,255,0), 2)

      # Find a suitable position to annotate the shape type
      text_x, text_y = approx[0][0][0], approx[0][0][1] - 10

      # Annotate the shape type
      cv2.putText(annotated_image, shape_type, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    p = 1
    for shape_type, distance in dist_list[counter1]:
      # Find a suitable position to annotate the distance
      text_x, text_y = 20, 20*p  # Adjust the position as needed
      p += 1
      # Annotate the distance
      cv2.putText(annotated_image, f"{shape_type}: {distance:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    img_name = "s_d_NF_" + str(counter1+1) + ".jpg"
    cv2.imwrite(os.path.join(save_dom1[4], img_name), annotated_image)
    counter1 += 1

  #DOM2
  ann_imgs2 = []

  for img in smooth2:
    ann = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ann_imgs2.append(ann)

  counter1 = 0
  for img in ann_imgs2:
    annotated_image = img.copy()

    for shape_type, cent, approx in roi_list[counter1]:
      # Draw the shape
      cv2.drawContours(annotated_image, [approx], -1, (0,255,0), 2)

      # Find a suitable position to annotate the shape type
      text_x, text_y = approx[0][0][0], approx[0][0][1] - 10

      # Annotate the shape type
      cv2.putText(annotated_image, shape_type, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    p = 1
    for shape_type, distance in distance_from_mid_list2[counter1]:
      # Find a suitable position to annotate the distance
      text_x, text_y = 20, 20*p  # Adjust the position as needed
      p += 1
      # Annotate the distance
      cv2.putText(annotated_image, f"{shape_type}: {distance:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    img_name = "s_d_FF_" + str(counter1+1) + ".jpg"
    cv2.imwrite(os.path.join(save_dom2[4], img_name), annotated_image )
    counter1 += 1

  print("Shapes Annotated")

  #annotate original image with distance lines (DOM1)
  neighbor_range1 = 90
  new_dom1_images = []
  data_dist1 = []


  for img in new_dom1:
    bgr_img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
    bgr_img = cv2.resize(bgr_img, ts1, interpolation=cv2.INTER_AREA)
    new_dom1_images.append(bgr_img)


  shapes_list = new_shapes1
  new_dom_imgs = new_dom1_images
  c_list = centroids_img_list1
  neighbor_range = neighbor_range1

  counter2 = 0
  for img in new_dom_imgs:
    annotated_image = img.copy()
    #print("new img", i)
    d_d = []
    for n in range(len(shapes_list[counter2])):
      for m in range(n+1,len(shapes_list[counter2])):
        #find distance
        d = np.sqrt((c_list[counter2][n][0][0] - c_list[counter2][m][0][0])**2 + (c_list[counter2][n][0][1] - c_list[counter2][m][0][1])**2)
        #print(d)
        if d <= neighbor_range:
          rand = (random.randint(50,255),random.randint(50,255),random.randint(50,255))
          #print("From: ", shapes_list[counter2][n][0], " To: ", shapes_list[counter2][m][0], " Distance : ", d)
          d_d.append((shapes_list[counter2][n][0],shapes_list[counter2][m][0],d))# (From, To, Distance)
          cv2.line(annotated_image, c_list[counter2][n][0], c_list[counter2][m][0], rand ,2)

          #calculate the midpoint of the line segment
          mp = ((c_list[counter2][n][0][0] + c_list[counter2][m][0][0])//2, (c_list[counter2][n][0][1] + c_list[counter2][m][0][1])//2)
          cv2.putText(annotated_image, f"{int(d)}", mp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rand , 2)

    #print(len(d_d))
    data_dist1.append(d_d)

    img_name = "annotated_NF_" + str(counter2+1) + ".jpg"
    cv2.imwrite(os.path.join(save_dom1[4], img_name), annotated_image)
    counter2 += 1

  #annotate original image with distance lines (DOM2)
  new_dom2_images = []
  data_dist2 = []

  for img in new_dom2:
    bgr_img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
    bgr_img = cv2.resize(bgr_img, ts2, interpolation=cv2.INTER_AREA)
    new_dom2_images.append(bgr_img)

  counter2 = 0
  for img in new_dom2_images:
    annotated_image = img.copy()
    #print("new img")
    d_d = []
    for n in range(len(roi_list[counter2])):
      for m in range(n+1,len(roi_list[counter2])):
        #find distance
        d = np.sqrt((roi_list[counter2][n][1][0] - roi_list[counter2][m][1][0])**2 + (roi_list[counter2][n][1][1] - roi_list[counter2][m][1][1])**2)
        #print(d)
        rand = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        #print("From: ", shapes_list[counter2][n][0], " To: ", shapes_list[counter2][m][0], " Distance : ", d)
        d_d.append((roi_list[counter2][n][0], roi_list[counter2][m][0],d))# (From, To, Distance)
        cv2.line(annotated_image, roi_list[counter2][n][1], roi_list[counter2][m][1], rand ,1)

        #calculate the midpoint of the line segment
        mp = ((roi_list[counter2][n][1][0] + roi_list[counter2][m][1][0])//2, (roi_list[counter2][n][1][1] + roi_list[counter2][m][1][1])//2)
        cv2.putText(annotated_image, f"{int(d)}", mp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rand , 1)
      cv2.putText(annotated_image, roi_list[counter2][n][0], (roi_list[counter2][n][1][0]-25,roi_list[counter2][n][1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2)
    data_dist2.append(d_d)
    img_name = "annotated_FF_" + str(counter2+1) + ".jpg"
    cv2.imwrite(os.path.join(save_dom2[4], img_name), annotated_image)
    counter2 += 1

  print("Original Images Annotated")

  # data augmentation for data recording

  aug1 = []
  aug2 = []

  counter4 = 0
  for i in range(l1):
      a = len(distance_from_mid_list1[counter4])
      b = len(data_dist1[counter4])
      temp_aug = []
      if a >= b:
          counter5 = 0
          for shape_type, distance in distance_from_mid_list1[counter4]:
              if counter5 >= b:
                  temp_aug.append((shape_type,f'{distance:.2f}', '', '', None))
              else:
                  temp_aug.append((shape_type,f'{distance:.2f}', data_dist1[counter4][counter5][0], data_dist1[counter4][counter5][1], f'{data_dist1[counter4][counter5][2]:.2f}'))
              counter5 += 1
      else:
          counter6 = 0
          for pack in data_dist1[counter4]:
              if counter6 >= a:
                  temp_aug.append(('', None, pack[0], pack[1], pack[2]))
              else:
                  temp_aug.append((distance_from_mid_list1[counter4][counter6][0], f'{distance_from_mid_list1[counter4][counter6][1]:.2f}', pack[0], pack[1], f'{pack[2]:.2f}'))
              counter6 += 1
      aug1.append(temp_aug)
      counter4 += 1

  counter4 = 0
  for i in range(l2):
      a = len(distance_from_mid_list2[counter4])
      b = len(data_dist2[counter4])
      temp_aug = []
      if a >= b:
          counter5 = 0
          for shape_type, distance in distance_from_mid_list2[counter4]:
              if counter5 >= b:
                  temp_aug.append((shape_type, f'{distance:.2f}', '', '', ''))
              else:
                  temp_aug.append((shape_type, f'{distance:.2f}', data_dist2[counter4][counter5][0], data_dist2[counter4][counter5][1], f'{data_dist2[counter4][counter5][2]:.2f}'))
              counter5 += 1
      else:
          counter6 = 0
          for pack in data_dist2[counter4]:
              if counter6 >= a:
                  temp_aug.append(('', '', pack[0], pack[1], f'{pack[2]:.2f}'))
              else:
                  temp_aug.append((distance_from_mid_list2[counter4][counter6][0], f'{distance_from_mid_list2[counter4][counter6][1]:.2f}', pack[0], pack[1], f'{pack[2]:.2f}'))
              counter6 += 1
      aug2.append(temp_aug)
      counter4 += 1

  # Step 6 Data Recording
  # Assuming 'distances' is a list of tuples (shape_type, distance)
  counter3 = 0
  for img in aug1:
    file_name = "Data_NF_" + str(counter3+1) + ".csv" # time stamp the folders in the future
    with open(os.path.join(save_dom1[5], file_name), 'w', newline='') as csvfile:
      fieldnames = ['Shape Type', 'Distance From Center', '','', 'From', 'To', 'Dist']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      #print("img")
      writer.writeheader()
      for shape_type, distance, f, t, d in img:
        #print("yes")
        row_data = {'Shape Type': shape_type, 'Distance From Center': distance, 'From': f, 'To': t, 'Dist': d}
        writer.writerow(row_data)
    counter3 += 1

  counter3 = 0
  for img in aug2:
    file_name = "Data_FF_" + str(counter3+1) + ".csv" # time stamp the folders in the future
    with open(os.path.join(save_dom2[5], file_name), 'w', newline='') as csvfile:
      fieldnames = ['Shape Type', 'Distance From Center', '','', 'From', 'To', 'Dist']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

      writer.writeheader()
      for shape_type, distance, f, t, d in img:
        row_data = {'Shape Type': shape_type, 'Distance From Center': distance, 'From': f, 'To': t, 'Dist': d}
        writer.writerow(row_data)
    counter3 += 1

  print("Data Recorded")

  # Stamped
  fp7 = 's7_Stamped/'
  folderL = os.path.join(f2,fp7)
  # Check if the folder exists, and if not, create it
  if not os.path.exists(folderL):
      os.makedirs(folderL)
  hc_images2 = []
  for img in new_dom2:
    bgr_img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
    bgr_img = cv2.resize(bgr_img, ts2, interpolation=cv2.INTER_AREA)
    hc_images2.append(bgr_img)

  counter2 = 0
  for img in hc_images2:
    annotated_image = img.copy()
    #d_d = []
    for i in range(4):
      if i == 0:
        hex = blue_hex
      elif i == 1:
        hex = blue_hex
      elif i == 2:
        hex = blue_hex
      else:
        hex = blue_hex
      for n in range(len(hex)):
        for m in range(n+1,len(hex)):
          #find distance
          d = np.sqrt((hex[n][0] - hex[m][0])**2 + (hex[n][1] - hex[m][1])**2)

          rand = (random.randint(0,255),random.randint(0,255),random.randint(10,255))
          #print("From: ", shapes_list[counter2][n][0], " To: ", shapes_list[counter2][m][0], " Distance : ", d)
          #d_d.append((new_shapes2[counter2][n][0],new_shapes2[counter2][m][0],d))# (From, To, Distance)
          cv2.line(annotated_image, hex[n], hex[m], rand ,1)

          #calculate the midpoint of the line segment
          mp = ((hex[n][0] + hex[m][0])//2, (hex[n][1] + hex[m][1])//2)
          cv2.putText(annotated_image, f"{int(d)}", mp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rand , 1)
    img_name = "annotated_FF_" + str(counter2+1) + ".jpg"
    cv2.imwrite(os.path.join(folderL, img_name), annotated_image)
    counter2 += 1

  print("Stamped Version DOM 2 Complete")

  hex_label = [0,1,2,3,4,5]
  # Pattern Notes

  ratio_b_list = []
  ratio_r_list = []
  ratio_o_list = []
  ratio_g_list = []
  #baseline = [86.53,86.53,100,86.53,86.53,100]
  baseline = []
  for coord in blue_hex:
      d = np.sqrt((coord[0] - center[0])**2 + (coord[1] - center[1])**2)
      baseline.append(d)
  for i in range(l2):
      ratio_b = []
      counter = 0
      for vertex, centr in b_hex[i]:
          ratio = 0
          if vertex == 1:
              b_line = np.sqrt((c_img[i][0] - centr[0])**2 + (c_img[i][1] - centr[1])**2)
              ratio = b_line/baseline[counter]
          ratio_b.append(f'{ratio:.2f}')
          counter += 1
      counter = 0
      ratio_r = []
      for vertex, centr in r_hex[i]:
          ratio = 0
          if vertex == 1:
              b_line = np.sqrt((c_img[i][0] - centr[0])**2 + (c_img[i][1] - centr[1])**2)
              ratio = b_line / baseline[counter]
          ratio_r.append(f'{ratio:.2f}')
          counter += 1
      counter = 0
      ratio_o = []
      for vertex, centr in o_hex[i]:
          ratio = 0
          if vertex == 1:
              b_line = np.sqrt((c_img[i][0] - centr[0])**2 + (c_img[i][1] - centr[1])**2)
              ratio = b_line / baseline[counter]
          ratio_o.append(f'{ratio:.2f}')
          counter += 1
      counter = 0
      ratio_g = []
      for j in range(6):
          ratio = 0
          if g_hex[i][j-2][0] == 1:
              b_line = np.sqrt((c_img[i][0] - g_hex[i][j-2][1][0])**2 + (c_img[i][1] - g_hex[i][j-2][1][0])**2)
              ratio = b_line / baseline[counter]
          ratio_g.append(f'{ratio:.2f}')
          counter += 1
      ratio_b_list.append(ratio_b)
      ratio_r_list.append(ratio_r)
      ratio_o_list.append(ratio_o)
      ratio_g_list.append(ratio_g)

  #Record the Ratio data
  # Assuming 'distances' is a list of tuples (shape_type, distance)
  counter3 = 0
  for i in range(l2):
    file_name = "Ratio_FF_" + str(counter3+1) + ".csv" # time stamp the folders in the future
    with open(os.path.join(save_dom2[5], file_name), 'w', newline='') as csvfile:
      fieldnames = ['Edge', 'Blue Ratio', '','', 'Green Ratio', 'Red Ratio', 'Outter Ratio']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for z in range(6):
        row_data = {'Edge': hex_label[z], 'Blue Ratio': ratio_b_list[i][z], 'Green Ratio': ratio_g_list[i][z], 'Red Ratio': ratio_r_list[i][z], 'Outter Ratio': ratio_o_list[i][z]}
        writer.writerow(row_data)
    counter3 += 1

  print("Ratio Files Created For DOM 2")