# -*- coding: utf-8 -*-
import cv2
import os
import math
import numpy as np

address = "/home/ray/Code/machine-learning/projects/capstone/data/svhn/full/test/"
base = "/home/ray/Code/Google-Street-View-House-Numbers-Digit-Localization/cascades/cascade"
outputFile = "./newCPUPython.txt"

#RATIOS = [1.0, 2.0]  # original ratios
MAX_NUM = 5
RATIOS = [1, 2]  # good for A111_1.jpg
#RATIOS = [0.5, 1]  # good for A203.jpg A206.jpg



class Box:
	def __init__(self, l, t, w, h):
		self.left = l;
		self.top = t;
		self.width = w;
		self.height = h;

	def area(self):
		return self.width * self.height

	def tl(self):
		return (self.left, self.top)

	def br(self):
		return (self.left+self.width, self.top+self.height)

	def right(self):
		return self.left+self.width

	def bottom(self):
		return self.top+self.height


def Overlap(box1, box2):
	# to be continued
	overlap = Box(0, 0, 0, 0)
	if box1.left+box1.width<box2.left or \
		box2.left+box2.width<box1.left:
		return overlap
	if box1.top+box1.height<box2.top or \
		box2.top+box2.height<box1.top:
		return overlap

	left = max(box1.left, box2.left)
	right = min(box1.left+box1.width, box2.left+box2.width)
	top = max(box1.top, box2.top)
	bottom = min(box1.top+box1.height, box2.top+box2.height)

	return Box(left, top, right-left, bottom-top)


def write(digit_list, imNo, outputFile):
	mini = 4
	f = open(outputFile, 'a')


	digit_len = len(digit_list)
	f.write(str(imNo)+' '+str(digit_len))

	for i in range(digit_len):
		format_str = ' ' + str(digit_list[i].left) + \
					 ' ' + str(digit_list[i].top) + \
					 ' ' + str(digit_list[i].width) + \
					 ' ' + str(digit_list[i].height)
		f.write(format_str)
	f.write('\n')
	f.close()

			

# The way of 'sigma' being computed can be replaced by the standard derivative function
def stats(numList):
	length = len(numList)
	sumq = np.sum(numList)
	sumsq = np.sum(np.square(numList))

	mu = sumq/length
	sigma = math.sqrt((sumsq/length)-(mu*mu))

	return mu, sigma



# @func: 使用正态分布将处于边缘的 height*width 的区域过滤掉
# @param:
#      all_digits -- 保存了所有检测到的数字区域的坐标（包括 false positive）
#      dist -- float，相当于正态分布的一个概率阈值
# @return: 根据概率分布过滤后的数字区域
def area_filter(all_digits, dist = 0.2):
	wTimesh = []
	for each_digit in all_digits:
		wTimesh.append(each_digit.width * each_digit.height)
	wTimesh = np.array(wTimesh)

	#mu = np.mean(wTimesh)
	#sigma = np.sum(wTimesh)
	mu, sigma = stats(wTimesh)

	filter_digits = []
	for each_digit in all_digits:
		if math.fabs(each_digit.height*each_digit.width - mu) <= (dist*sigma+25):
			filter_digits.append(each_digit)

	return filter_digits


# @func: 所有区域按重叠程度合并（聚簇），合并条件为　重叠区域面积达到原始区域面积一半
# @param: 
#      filter_digits -- 使用概率分布过滤掉一些异常区域后的数字矩阵
# @return: 
#	   combined_digits --　合并后的区域列表
#	   confidence -- 合并次数组成列表，每个元素对应 combined_digits　的一个区域。 
#					　一个区域合并的次数越多，存在数字的可能越低
def cluster(filter_digits):
	cc = 1.0
	combined_digits = []
	confidence = []
	if len(filter_digits) == 0:
		print('Filtered all digits area!!')
		return combined_digits, confidence

	tmp = filter_digits[0]
	for i in range(1, len(filter_digits)):
		# Unimplemented function
		overlap = Overlap(filter_digits[i], tmp)

		# overlap either half area
		if (overlap.area()>0.5*tmp.area()) or (overlap.area()>0.5*filter_digits[i].area()):
			# recompute the average coordinate
			avg_left = (tmp.left*cc+filter_digits[i].left)//(cc+1)
			avg_top = (tmp.top*cc+filter_digits[i].top)//(cc+1)
			avg_right= ((tmp.left+tmp.width)*cc+(filter_digits[i].left+filter_digits[i].width)) // (cc+1)
			avg_bottom = ((tmp.top+tmp.height)*cc+(filter_digits[i].top+filter_digits[i].height)) // (cc+1)

			# left, top, width, height
			tmp = Box(int(avg_left), int(avg_top), int(avg_right-avg_left), int(avg_bottom-avg_top))
			cc = cc+1
		else:
			combined_digits.append(tmp)
			tmp = filter_digits[i]
			
			# 越多区域被合并，该区域的 confidence 就越大
			confidence.append(cc)
			cc = 1

	combined_digits.append(tmp)
	confidence.append(cc)

	return combined_digits, confidence


# @func: 将 cluster 之后的所有区域按照 confidence 的值从小到大排序
# @param: digits -- 合并之后的可能存在数字的区域
#		  confidence -- 重合率，每个元素与 digits 的每个元素对应。
#						重合率越高代表对应的 digit 区域合并了越多的矩阵
#		  max_num -- 允许存在的最大数字个数
# @return: 出现概率最大的几个（不大于max_num个）数字区域
def sortByConfidence(digits, confidence, max_num):
	# 按照 confidence 的值从小到大排序
	# 按照实验结果来看，从小到大的排序确实能输出更精确的结果
	for i in range(len(digits)):
		for j in range(len(digits)):
			if confidence[j] < confidence[i]:
				digits[i], digits[j] = digits[j], digits[i]
				confidence[i], confidence[j] = confidence[j], confidence[i]
	length = min(len(digits), max_num)
	
	ret_digits = [digits[i] for i in range(length)]
	return ret_digits



def getDigitArea(cascade_list, img, enlarge, ratios):
	tmp_copy = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_scale = []

	all_digits = []
	for k in range(len(ratios)):
		# y 轴没进行 ratio 缩放
		tmp_img = cv2.resize(img, (0, 0), fx=ratios[k]*enlarge, fy=enlarge)
		img_scale.append(tmp_img)

		for i in range(10):
			#digits = cascade_list[i].detectMultiScale(img_scale[k], 1.1, 3, 
			#	0|cv2.CV_HAAR_SCALE_IMAGE, [20,30], [400, 600])
			digits = cascade_list[i].detectMultiScale(img_scale[k], 1.1, 3,
				0, (20,30), (400, 600))

			for x,y,width,height in digits:
				curr_x = x // ratios[k]*enlarge
				curr_y = y // enlarge  		# y 轴没进行 ratio 缩放
				curr_width = width // (ratios[k]*enlarge);
				curr_height = height // enlarge;
				all_digits.append(Box(int(curr_x), int(curr_y), \
									int(curr_width), int(curr_height)))
	#　按每个　area　的 x　轴的中心点排序
	all_digits = sorted(all_digits, key=lambda each_digit : (each_digit.left+each_digit.width/2))

	if(len(all_digits) > 0):
		filter_digits = area_filter(all_digits)
		cluster_digit, confidence = cluster(filter_digits)
		ret_digits = sortByConfidence(cluster_digit, confidence, MAX_NUM)
		return ret_digits

	else:
		# Have probability to cause infinite recursion
		return getDigitArea(cascade_list, tmp_copy, 2*enlarge, ratios)


def localize(filepath, ratios):
	digit_cascade = []
	for i in range(10):
		filename = base+str(i)+'/cascade.xml'
		digit_cascade.append(cv2.CascadeClassifier(filename))
	print('digit_cascade constructed.')

	img = cv2.imread(filepath)
	return getDigitArea(digit_cascade, img, 1, ratios)


def main():
	digit_cascade = []
	for i in range(10):
		filename = base+str(i)+'/cascade.xml'
		digit_cascade.append(cv2.CascadeClassifier(filename))
	print('digit_cascade constructed.')

	try:
		os.remove(outputFile)
	except:
		pass

	for i in range(1, 101):
		print('Being localize...')
		filename = address + str(i) + '.png'
		img = cv2.imread(filename, 1)
		ret_digit = getDigitArea(digit_cascade, img, 1, RATIOS)
		write(ret_digit, i+1, outputFile)


if __name__ == '__main__':
	main()