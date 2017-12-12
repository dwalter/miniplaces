from PIL import Image
import os

import cv2
import numpy as np

import skimage
from skimage import color
 
def black_and_white(input_image_path, output_image_path):
   color_image = Image.open(input_image_path)
   bw = color_image.convert('L')
   bw.save(output_image_path)



# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
def image_rgb2lab(input_image_path, output_image_path):
  image_rgb = cv2.imread(input_image_path) / 255.

  # Convert BGR to HSV
  # image_lab = cv2.cvtColor(image_rgb, 44) 
  image_lab = skimage.color.rgb2lab(image_rgb)
  # 45 is the constant for rgb2lab, 44 is the const. for gbr2lab
  # https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gga4e0972be5de079fed4e3a10e24ef5ef0a860b72b4741c431e81340fafef5eca24&gsc.tab=0
  
  cv2.imwrite(output_image_path, image_lab)

def image_lab2rgb(input_image_path, output_image_path):
  image_lab = cv2.imread(input_image_path)

  # Convert BGR to HSV
  image_rgb = cv2.cvtColor(image_lab, 56) 
  # cv::COLOR_Lab2RGB = 57
  # cv::COLOR_Lab2BGR = 56
  # https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gga4e0972be5de079fed4e3a10e24ef5ef0a860b72b4741c431e81340fafef5eca24&gsc.tab=0
  
  cv2.imwrite(output_image_path, image_rgb)


def main():
  func_to_run = image_rgb2lab
  # func_to_run = image_lab2rgb
  data_dir = '/Users/dwalter/Classes/6.819/finalProject/'

  pics_open_dir = data_dir + 'chaplin_bw'
  # pics_open_dir = data_dir + 'chaplin1_frames_lab'
  pics_save_dir = data_dir + 'chaplin1_frames_lab'
  # pics_save_dir = data_dir + 'chaplin1_frames_lab_to_rgb'
  # print 'here1'
  for filename in os.listdir(pics_open_dir):
      print filename
      if filename.endswith(".jpg"):
          # print 'here'
          open_path = pics_open_dir + '/' + filename
          save_path = pics_save_dir + '/' + open_path[-9:]
          func_to_run(open_path, save_path)

 
if __name__ == '__main__':
  main()
    
