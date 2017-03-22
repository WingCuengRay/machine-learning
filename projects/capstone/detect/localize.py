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


# @func: 合并所有的数字区域
# @para: digit_boxs - 由一个或多个数字区域(Box)组成的列表
# @return: 合并后的一个大的包含全部数字的矩形区域
def mergeAllAreas(digit_boxes):
	min_left = 99999
	min_top = 99999
	max_right = 0
	max_bottom = 0
	for each_box in digit_boxes:
		if min_left > each_box.left:
			min_left = each_box.left
		if min_top > each_box.top:
			min_top = each_box.top

		if max_right < each_box.right():
			max_right =  each_box.right()
		if max_bottom < each_box.bottom():
			max_bottom = each_box.bottom()

	return Box(min_left, min_top, max_right-min_left, max_bottom-min_top)



def main():
	if len(sys.argv) != 2:
		print('Format: python localizate.py filePath')
		exit()

	ret_digit = localize(sys.argv[1], RATIOS)
	#ret_area = mergeAllAreas(ret_digit)
	#show_image(sys.argv[1], [ret_area])
	show_image(sys.argv[1], ret_digit)
	
	
if __name__ == '__main__':
	main()