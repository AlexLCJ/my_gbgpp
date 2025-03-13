import copy
import csv
import os
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
import warnings
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

class GranularBall:
	"""class of the granular ball"""

	def __init__(self, data, attribute):
		"""
		:param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
		and each of the preceding columns corresponds to a feature
		"""
		self.data = data[:, :] # all rows and all cols
		self.attribute = attribute
		self.data_no_label = data[:, attribute] # 属性列
		self.num, self.dim = self.data_no_label.shape # num:行数，dim：列数
		self.center = self.data_no_label.mean(0) # 粒球的中心点，属性列的平均值作为中心点的维度
		self.label, self.purity, self.r = self.__get_label_and_purity_and_r() # label, purity, radius

	def __get_label_and_purity_and_r(self):
		"""
		:return: the label and purity of the granular ball.
		"""
		count = Counter(self.data[:, -2]) # 统计每类的数量，the "-2" column is the class label

		label = max(count, key=count.get)  # count中值最大的键，用来代表这个粒球
		purity = count[label] / self.num  # 粒球的纯度： label占据的比例
		# 计算半径
		arr = np.array(self.data_no_label) - self.center  # 每个样本相对于中心点的偏差
		ar = np.square(arr)
		a = np.sqrt(np.sum(ar, 1)) # 每个样本到中心点的距离
		r = max(a)  # ball's radius is max disitient
		return label, purity, r

	def split_2balls(self):
		"""
	    split into two balls
	    This method attempts to divide the current dataset into two subsets (balls) based on the data's characteristics.
    	It uses the KMeans algorithm to determine the division.
    	If the data cannot be effectively split into two clusters,it returns the original data as a single ball.
    	Returns:
        	A list containing one or two GranularBall objects, representing the split data subsets.
		"""
		# label_cluster = KMeans(X=self.data_no_label, n_clusters=2)[1]
		# print(self.data_no_label.shape,self.num,self.dim)#
		labs = set(self.data[:, -2].tolist()) # class label（set去重）
		# print("labs",labs)
		i1 = 9999 # 初始化最小值
		ti1 = -1 # 初始化索引
		ti0 = -1 # 初始化索引
		Balls = [] # 分割后的粒球
		i0 = 9999 # 初始化最小值
		dol = np.sum(self.data_no_label, axis=1) # 每个样本（横向求和）的特征数总和
		if len(labs) > 1: # 多个类别
			for i in range(len(self.data)):
				if self.data[i, -2] == 1 and dol[i] < i1: # 类别为1且且特征数总和小于i1
					i1 = dol[i] # 记录最小值
					ti1 = i # 索引
				elif self.data[i, -2] != 1 and dol[i] < i0: # 类别不为1且特征数总和小于i0
					i0 = dol[i] # 最小值
					ti0 = i # 索引
			ini = self.data_no_label[[ti0, ti1], :] # 构造初始聚类中心，选择了这两个点
			clu = KMeans(n_clusters=2, init=ini).fit(self.data_no_label)  # select primary sample center
			label_cluster = clu.labels_ # 获取聚类标签
			if len(set(label_cluster)) > 1: # 聚类结果中有两个不同的标签
				ball1 = GranularBall(self.data[label_cluster == 0, :], self.attribute) # 第一个球
				ball2 = GranularBall(self.data[label_cluster == 1, :], self.attribute) # 第二个球
				Balls.append(ball1)
				Balls.append(ball2)
			else:
				Balls.append(self)
		else:
			Balls.append(self)
		return Balls


def funtion(ball_list, minsam):
	"""
	处理粒球之间的重叠问题
	"""
	Ball_list = ball_list # 粒球列表
	Ball_list = sorted(Ball_list, key=lambda x: -x.r, reverse=True) # 排序，根据粒球半径从大到小
	ballsNum = len(Ball_list)
	j = 0
	ball = []
	while True:
		if len(ball) == 0:
			# 依次添加粒球的中心、半径、标签、样本数
			ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num])
			j += 1
		else:
			flag = False
			for index, values in enumerate(ball):
				# 检查当前粒球与已处理粒球是否标签不同且相交，并且满足最小样本数条件
				if values[2] != Ball_list[j].label and (
						np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) < (
						values[1] + Ball_list[j].r) and Ball_list[j].r > 0 and Ball_list[j].num >= minsam / 2 and \
						values[3] >= minsam / 2:
					# 尝试将当前粒球分裂成两个粒球
					balls = Ball_list[j].split_2balls()
					if len(balls) > 1:
						# 分裂成功，更新Ball_list
						Ball_list[j] = balls[0]
						Ball_list.append(balls[1])
						ballsNum += 1
					else:
						Ball_list[j] = balls[0]
			if flag == False:
				# print(8)
				# 当前粒球没有与其他粒球重叠或不满足分裂条件，将其添加到ball列表中
				ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num])
				j += 1
		if j >= ballsNum:
			break
	# ball = [] # 是否需要？？
	# j = 0
	#if two ball's label is different and overlapped，in this step,we can continut split positive domain can keep boundary region don't change, but this measure is unnecessary
	# while 1:
	# 	if len(ball) == 0:
	# 		ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num,Ball_list[j]])
	# 		j += 1
	# 	else:
	# 		flag = False
	# 		for index, values in enumerate(ball):
	# 			if values[2] != Ball_list[j].label and (
	# 					np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) < (
	# 					values[1] + Ball_list[j].r) and Ball_list[j].r > 0 and (Ball_list[j].purity>0.995 or values[-1].purity>0.995):
	# 				if(values[-1].purity<0.99):
	# 					balls = Ball_list[j].split_2balls()
	# 					if len(balls) > 1:
	# 						Ball_list[j] = balls[0]
	# 						Ball_list.append(balls[1])
	# 						ballsNum += 1
	# 					else:
	# 						Ball_list[j] = balls[0]
	# 				elif Ball_list[j].purity<0.99:
	# 					balls = values[-1].split_2balls()
	# 					if len(balls) > 1:
	# 						values[-1] = balls[0]
	# 						Ball_list.append(balls[1])
	# 						ballsNum += 1
	# 					else:
	# 						Ball_list[j] = balls[0]
	# 		if flag == False:
	# 			# print(8)
	# 			ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num,Ball_list[j]])
	# 			j += 1
	# 	if j >= ballsNum:
	# 		break
	return Ball_list


def overlap_resolve(ball_list, data, attributes_reduction, min_sam):
	"""
	通过KMeans聚类来解决粒球之间的重叠问题
	:param ball_list: 粒球列表
    :param data: 数据集
    :param attributes_reduction: 属性约简后的属性索引
    :param min_sam: 最小样本数阈值
    :return: 更新后的粒球列表
	"""
	Ball_list = funtion(ball_list, min_sam)  # conitinue to split ball which are overlapped
	# do last overlap for granular ball aimed raise ball's quality
	while True:
		init_center = []  # ball's center
		Ball_num1 = len(Ball_list)
		for i in range(len(Ball_list)):
			init_center.append(Ball_list[i].center)
		# 聚类中心为init_center , n_clusters=len(Ball_list)
		ClusterLists = KMeans(init=np.array(init_center),
							  n_clusters=len(Ball_list)).fit(data[:, attributes_reduction])

		data_label = ClusterLists.labels_
		ball_list = []
		for i in set(data_label):
			ball_list.append(GranularBall(data[data_label == i, :], attributes_reduction))
		Ball_list = funtion(ball_list, min_sam)
		Ball_num2 = len(Ball_list)  # get ball numbers
		if Ball_num1 == Ball_num2:  # stop until ball's numbers don't change
			break
	return Ball_list


class GBList:
	"""
	class of the list of granular ball
	"""

	def __init__(self, data=None, attribu=[]):
		"""
		:param data:None
		:param attribu:属性列表,默认为空
		"""
		self.data = data[:, :] # data[:,:]浅拷贝
		self.attribu = attribu # 属性列表
		self.granular_balls = [GranularBall(self.data, self.attribu)]  # gbs is initialized with all data

	def init_granular_balls(self, purity=0.996, min_sample=1):
		"""
		Split the balls, initialize the balls list.
		purty=1,min_sample=2d
		:param purity: If the purity of a ball is greater than this value, stop splitting. (人工设置的超参数？)
		:param min_sample: If the number of samples of a ball is less than this value, stop splitting.(保证最小粒球内的样本)
		"""
		ll = len(self.granular_balls) # 粒球的个数
		i = 0 # 初始索引
		while True:
			# 如果当前粒球的纯度低于阈值且样本数量大于最小样本数，则分裂粒球
			if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
				split_balls = self.granular_balls[i].split_2balls() # 分裂球，分裂成两个
				if len(split_balls) > 1:
					# 分裂成功后：第一个球替换掉当前球，并加入新的球，此时ll+1
					self.granular_balls[i] = split_balls[0]
					self.granular_balls.append(split_balls[1])
					ll += 1
				else:
					# 索引加一，往后查看一位
					i += 1
			else:
				i += 1
			if i >= ll:
				break
		ball_lists = self.granular_balls  # 获取当前粒球列表
		Bal_List = overlap_resolve(ball_lists, self.data, self.attribu, min_sample)  # do overlap 解决重叠问题
		self.granular_balls = Bal_List # 更新粒球列表
		self.get_data() # 所有粒球数据
		self.data = self.get_data() # 更新数据

	def get_data_size(self):
		"""
		return: list:每个粒球数据的大小（data长度为参考）
		"""
		return list(map(lambda x: len(x.data), self.granular_balls))

	def get_purity(self):
		"""
		return: list:所有粒球的纯度
		"""
		return list(map(lambda x: x.purity, self.granular_balls))

	def get_center(self):
		"""
		:return: the center of each ball.
		"""
		return np.array(list(map(lambda x: x.center, self.granular_balls)))

	def get_r(self):
		"""
		:return: return radius r
		"""
		return np.array(list(map(lambda x: x.r, self.granular_balls)))

	def get_data(self):
		"""
		:return: Data from all existing granular balls in the GBlist.
		"""
		list_data = [ball.data for ball in self.granular_balls]
		# np.vstack()表示堆叠数组，将所有球内的数据总和在一起
		return np.vstack(list_data)

	def del_ball(self, purty=0., num_data=0):
		"""
		:param purty:纯度阈值
		:param num_data:数据量阈值
		"""
		# delete ball
		T_ball = []
		for ball in self.granular_balls:
			if ball.purity >= purty and ball.num >= num_data:
				T_ball.append(ball)
		self.granular_balls = T_ball.copy()
		self.data = self.get_data()

	def R_get_center(self, i):
		# get ball's center
		attribu = self.attribu
		attribu.append(i)
		centers = []
		for ball in range(self.granular_balls):
			center = []
			data_no_label = ball.data[:, attribu]
			center = data_no_label.mean(0)
			centers.append(center)
		return centers


def attribute_reduce(data, pur=1, d2=2):
	"""
	通过逐步添加属性并评估其对粒球纯度的影响，最终返回一个最优的属性列表
	:param data: dataset
	:param pur: purity threshold
	:param d2: min_samples, the default value is 2, 控制最小样本数量
	:return: reduction attribute
	"""
	bal_num = -9999
	attribu = []
	re_attribu = [i for i in range(len(data[0]) - 2)] #候选的属性，获取所有候选属性的索引列表,除去label和index
	while len(re_attribu):
		N_bal_num = -9999 # 初始化当前轮次的最大正域样本数量
		N_i = -1 # 初始化当前轮次的最佳属性索引
		N_attribu = copy.deepcopy(attribu) # 浅拷贝当前选择的属性
		for i in re_attribu: # 遍历所有候选属性
			N_attribu = copy.deepcopy(attribu) # 当前属性列表
			N_attribu.append(i) # 添加当前属性
			gb = GBList(data, N_attribu)  # create the list of granular balls
			gb.init_granular_balls(purity=pur, min_sample=2 * (len(data[0]) - d2))  # initialize the list
			ball_list1 = gb.granular_balls
			# for bal in ball_list1:
			# 	if bal.purity<=0.999:
			# 		print(bal.center,bal.r)
			# 	else:
			# 		print(bal.center,bal.r,bal.label)
			Pos_num = 0
			for ball in ball_list1:
				if ball.purity >= 1:
					Pos_num += ball.num  # find the current  domain samples
			if Pos_num > N_bal_num:
				N_bal_num = Pos_num
				N_i = i
		if N_bal_num >= bal_num:
			bal_num = N_bal_num
			attribu.append(N_i)
			re_attribu.remove(N_i)
		else:
			return attribu
	return attribu


def mean_std(a):
	# calucate average and standard
	a = np.array(a)
	std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
	return a.mean(), std


if __name__ == "__main__":
	datan = ["wine"]

	for name in datan:
		# 打开结果文件以写入模式
		with open(
				r"C:\Users\李昌峻\Desktop\粗糙集\GranularBall\GBRS A Unified Granular-ball Learning Model of Pawlak Rough Set and Neighborhood Rough Set\GBRS\Result\\"
				+ name + ".csv", "w", newline='', encoding="utf-8") as jg:
			writ = csv.writer(jg)
			df = pd.read_csv(
				r"C:\Users\李昌峻\Desktop\粗糙集\GranularBall\GBRS A Unified Granular-ball Learning Model of Pawlak Rough Set and Neighborhood Rough Set\GBRS\DataSet\\"
				+ name + ".csv")
			data = df.values
			numberSample, numberAttribute = data.shape
			# 归一化
			minMax = MinMaxScaler()
			U = np.hstack(
				(minMax.fit_transform(data[:, 1:]), data[:, 0].reshape(numberSample, 1)))  # 对特征部分进行归一化，并将类别标签拼接回数组末尾
			C = list(np.arange(0, numberAttribute - 1))  # 创建属性列索引列表
			D = list(set(U[:, -1]))  # 获取类别标签集合
			index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # 创建样本索引列
			sort_U = np.argsort(U[:, 0:-1], axis=0)  # 按照特征列排序
			U1 = np.hstack((U, index))  # 将索引列拼接到归一化后的数据末尾
			index = np.array(range(numberSample)).reshape(numberSample, 1)  # 再次创建样本索引列
			data_U = np.hstack((U, index))  # 将索引列拼接到归一化后的数据末尾
			purty = 1  # 设置粒球纯度阈值
			clf = KNeighborsClassifier(n_neighbors=5)
			orderAttributes = U[:, -1]  # 提取类别标签列
			mat_data = U[:, :-1]  # 提取特征矩阵
			# 初始化最大平均准确率、标准差和最佳属性组合
			maxavg = -1
			maxStd = 0
			maxRow = []
			for i in range((int)(numberAttribute)):
				nums = i
				Row = attribute_reduce(data_U, pur=purty, d2=nums)  # 进行属性约简，得到当前属性组合
				writ.writerow(["FGBNRS", Row])  # 将属性组合写入结果文件
				print("Row:", Row)  # 打印当前属性组合
				mat_data = U[:, Row]  # 根据当前属性组合提取特征矩阵
				scores = cross_val_score(clf, mat_data, orderAttributes, cv=5)  # 使用交叉验证计算分类器的准确率
				avg, std = mean_std(scores)  # 计算准确率的平均值和标准差
				# 更新最大平均准确率、标准差和最佳属性组合
				if maxavg < avg:
					maxavg = avg
					maxStd = std
					maxRow = copy.deepcopy(Row)
			# 打印最高平均准确率
			print("pre", maxavg)
			# 打印最佳属性组合
			print("row", maxRow)

