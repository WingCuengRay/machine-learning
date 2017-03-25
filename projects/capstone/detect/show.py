# -*- coding:utf8 -*-
from detect import localize
from detect import Box
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys
import os

#RATIOS = [1.0, 2.0]  # original ratios
#RATIOS = [1, 2]  # good for A111_1.jpg
RATIOS = [0.5, 1]  # good for A203.jpg A206.jpg
#RATIOS = [0.5]	#勉强可以: A203.jpg A206.jpg

# func: 在指定的画布上画一个矩形
# param: 
#    ax --  显示的画布
#    boxes --  一维数组，其元素类型为 NumberBox，该类型包含的矩形的坐标与长宽
def drawRectangle(ax, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        ax.add_patch(
            patches.Rectangle(
                (box.left, box.top),
                box.width,
                box.height,
                color = 'green',
                fill = False
            )
        )


def show_image(imgfile, digit_area):
	img = Image.open(imgfile)

	plt.close('all')
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1);
	drawRectangle(ax, digit_area)
	ax.imshow(img)
	plt.show()




def main():
	if len(sys.argv) != 2:
		print('Format: python localizate.py filePath')
		exit()

	ret_digit = localize(sys.argv[1], RATIOS)
	show_image(sys.argv[1], ret_digit)
	
	
if __name__ == '__main__':
	main()