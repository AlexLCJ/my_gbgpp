import csv
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
import warnings
from itertools import chain
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# A=[[2,3,4],[1,2,1]]

class GranularBall:
    def __init__(self, data, attributes):
        """
        the '-2' clo is the label, and the last is the index(added in the main function, data_U)
        """
        self.data = data
        self.attributes = attributes  # 特征列索引
        self.data_no_label = data[:, attributes]  # 属性列
        self.center =  self.data_no_label.mean(0)
        self.label, self.purity, self.radius = self._calculate_metadata()
        self.num_samples = len(data)
        self.children = []

    def calculate_center(self):
        return np.mean(self.data_no_label, axis=0)

    def _calculate_metadata(self):
        labels = self.data[:, -2]  # 标签在倒数第二列
        # print("_calculate_metadata",labels)
        label_counts = Counter(labels)
        majority_label = max(label_counts, key=label_counts.get)
        purity = label_counts[majority_label] / len(labels)
        # print("_calculate_metadata_purity",purity)
        distances = np.linalg.norm(self.data[:, self.attributes] - self.center, axis=1)
        radius = np.mean(distances)
        return majority_label, purity, radius

    def GBGpp_split(self, purity_threshold):
        if self.purity >= purity_threshold or self.num_samples == 1:
            return [self]

        # 获取多数类样本
        majority_mask = self.data[:, -2] == self.label  # 倒数第二列是label列
        majority_data = self.data[majority_mask] # lhg(T)

        if len(majority_data) == 0:
            return [self] # list

        new_center = np.mean(majority_data[:, self.attributes], axis=0)
        distances = np.linalg.norm(majority_data[:, self.attributes] - new_center, axis=1)
        new_radius = np.mean(distances)
        # 划分样本
        all_distances = np.linalg.norm(self.data[:, self.attributes] - new_center, axis=1)
        in_ball_mask = all_distances <= new_radius
        child_data = self.data[in_ball_mask]
        remaining_data = self.data[~in_ball_mask]
        # 创建子粒球
        child_ball = GranularBall(child_data, self.attributes)
        child_balls = [child_ball]
        if len(remaining_data) > 0:
            remaining_ball = GranularBall(remaining_data, self.attributes)
            child_balls += remaining_ball.GBGpp_split(purity_threshold)
        # 过滤
        filtered = []
        for b in child_balls:
            if b.num_samples > 1 or (b.num_samples == 1 and b.purity == 1.0):
                filtered.append(b)
        return filtered

    def split_using_DBGBC(self, eps_ratio=0.5, min_samples=2):
        # 处理半径为0的特殊情况
        if self.radius == 0:
            # print("Warning: Radius is zero, skip splitting")
            return [self]

        # 计算eps并确保其有效性
        eps = self.radius * eps_ratio
        eps = max(eps, 1e-7)
        try:
            # DBSCAN聚类
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.data_no_label)
            labels = db.labels_
            # 处理噪声点和单簇情况
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            n_clusters = len(unique_labels)
            if n_clusters < 2:
                return [self]

            # 使用K-means进行二次分割
            kmeans = KMeans(n_clusters=2, n_init=10).fit(self.data_no_label)
            mask = kmeans.labels_ == 0
            child1 = GranularBall(self.data[mask], self.attributes)
            child2 = GranularBall(self.data[~mask], self.attributes)

            # 递归
            return child1.split_using_DBGBC(eps_ratio, min_samples) + \
                child2.split_using_DBGBC(eps_ratio, min_samples)

        except Exception as e:
            print(f"Splitting failed: {str(e)}")
            return [self]

class GBGPlusPlus:
    def __init__(self, purity_threshold=1.0):
        self.purity_threshold = purity_threshold
        self.balls = []

    def fit(self, data, attributes):
        initial_ball = GranularBall(data, attributes)
        self.balls = self._split_recursive([initial_ball])
        self.balls = self._resolve_conflicts(self.balls)
        return self

    def _split_recursive(self, balls): # 分割粒球
        new_balls = []
        changed = False
        for ball in balls:
            if ball.purity < self.purity_threshold and ball.num_samples > 1:
                children = ball.GBGpp_split(self.purity_threshold)
                if len(children) > 1:
                    new_balls.extend(children)
                    changed = True
                else:
                    new_balls.append(ball)
            else:
                new_balls.append(ball)
        if changed:
            return self._split_recursive(new_balls)
        return new_balls

    def _resolve_conflicts(self, balls): # 解决冲突
        merged = []
        used = set()
        for i in range(len(balls)):
            if i in used:
                continue
            ball1 = balls[i]
            conflict = False
            for j in range(i + 1, len(balls)):
                if j in used:
                    continue
                ball2 = balls[j]
                if self._is_conflict(ball1, ball2):
                    merged_data = np.vstack([ball1.data, ball2.data]) # 先重合再继续分割
                    merged_ball = GranularBall(merged_data, ball1.attributes)
                    merged.append(merged_ball)
                    used.update([i, j])
                    conflict = True
                    break
            if not conflict:
                merged.append(ball1)
                used.add(i)
        # 添加未处理的球
        for j in range(len(balls)):
            if j not in used:
                merged.append(balls[j])

        return merged

    def _is_conflict(self, ball1, ball2): # 冲突判断
        if ball1.label == ball2.label:
            return False
        center_dist = np.linalg.norm(ball1.center - ball2.center)
        radius_diff = abs(ball1.radius - ball2.radius)
        return center_dist <= radius_diff # 如果中心距离小于等于半径差，则返回True，表示冲突


    def apply_DBGBC(self,eps_ratio=0.5,min_samples=2):
        "dbgbc优化方法，适用在GBGPP后直接.apply该方法"
        if not self.balls:
            print("Warning: Empty ball list")
            return

        self.balls = self._recursive_dbgbc_split(self.balls, eps_ratio, min_samples)
        return self

    def _recursive_dbgbc_split(self, balls, eps_ratio, min_samples):
        "使用dbgbc优化"
        if not balls:
            return []

        new_balls = []
        for ball in balls:
            children = ball.split_using_DBGBC(eps_ratio, min_samples)
            if len(children) == 1:
                new_balls.append(children[0])
            else:
                new_balls.extend(children)

        # 防止单层分割后列表过长
        if len(new_balls) == len(balls):
            return new_balls

        return self._recursive_dbgbc_split(new_balls, eps_ratio, min_samples)


class GBKNNPlusPlus:
    def __init__(self, purity_threshold=1.0):
        self.gbg = GBGPlusPlus(purity_threshold)

    def fit(self, X, y):
        data = np.hstack([X, y.reshape(-1, 1), np.arange(len(X)).reshape(-1, 1)])
        self.balls = self.gbg.fit(data, list(range(X.shape[1]))).balls # 调用对data进行拟合并保存结果到self.balls
        return self

    def predict(self, X_test):
        predictions = []
        total_samples = sum([b.num_samples for b in self.balls])

        for x in X_test:
            min_dist = float('inf')
            pred_label = None

            for ball in self.balls:
                # 计算调和距离
                # 通过从样本到球中心的欧几里得距离中减去球的样本数量与总样本数量的比率来计算
                dist = np.linalg.norm(x - ball.center)
                harmonic_dist = dist - (ball.num_samples / total_samples)
                # 对于每个样本，选择调和距离最小的球，并将该球的标签分配给样本
                if harmonic_dist < min_dist:
                    min_dist = harmonic_dist
                    pred_label = ball.label

            predictions.append(pred_label)
        return np.array(predictions)

def granular_ball_features(X, granular_balls,metric='euclidean'): # 这部分还在实验
    centers = np.array([ball.center for ball in granular_balls])
    # features = np.sqrt(np.sum((X[:, None] - centers) ** 2, axis=2))
    # return features

    if metric == 'euclidean':
        # 欧氏距离矩阵
        features = np.sqrt(np.sum((X[:, None] - centers) ** 2, axis=2))
    elif metric == 'cosine':
        # 余弦相似度矩阵
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        centers_norm = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        cosine_sim = np.dot(X_norm, centers_norm.T)
        features = 1 - cosine_sim
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return features

def evaluate_model(X_train, X_test, y_train, y_test):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1": f1_score(y_test, y_pred, average='weighted')
        }
    return results

if __name__ == "__main__":
    # 配置参数
    data_name = ['wine','zoo','horse','german','anneal','mushroom']
    # data_name = ['wine','credit']
    for name in data_name:
        with open(
                r"C:\Users\李昌峻\Desktop\粗糙集\GranularBall\GBRS A Unified Granular-ball Learning Model of Pawlak Rough Set and Neighborhood Rough Set\GBRS\Result\\"
                + name + ".csv", "w", newline='', encoding="utf-8"
        ) as jg:
            writ = csv.writer(jg)
            df = pd.read_csv(r"C:\Users\李昌峻\Desktop\粗糙集\GranularBall\dataset\\"
                    + name + ".csv")
            data = df.values
            numberSample, numberAttribute = data.shape
            minMax = MinMaxScaler()
            U = np.hstack(
                (minMax.fit_transform(data[:, 1:]), data[:, 0].reshape(numberSample, 1)))  # 归一化，类别标签拼接回数组末尾
            C = list(np.arange(0, numberAttribute - 1))  # 创建属性列索引列表
            D = list(set(U[:, -1]))  # 获取类别标签集合
            index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # 创建样本索引列
            sort_U = np.argsort(U[:, 0:-1], axis=0)  # 按照特征列排序
            U1 = np.hstack((U, index))  # 将索引列拼接到归一化后的数据末尾
            index = np.array(range(numberSample)).reshape(numberSample, 1)  # 再次创建样本索引列
            data_U = np.hstack((U, index))  # 将索引列拼接到归一化后的数据末尾
            # purity = 1  # 设置粒球纯度阈值
            orderAttributes = U[:, -1]  # 提取类别标签列
            mat_data = U[:, :-1]  # 提取特征矩阵
            # print(mat_data.shape) # (177,13)
            # print(U.shape)  # (177,14)
            # print(data_U.shape)  # (177,15)

            # ========================训练机器学习模型==========================

            # 训练集和测试集表示
            #X = mat_data # 特征列
            print(f"\n{'=' * 40}")
            print(f"数据集: {name}")
            print(f"{'=' * 40}")
            y = orderAttributes # 标签
            X = minMax.fit_transform(mat_data) # 特征
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # print("属性",list(range(X_train.shape[1])))

            gbg_start_time = time.time()  # 粒球生成计时开始
            gbg_plus_plus = GBGPlusPlus(purity_threshold=1.00)  # 实例
            gbg_plus_plus.fit(data_U, list(range(X_train.shape[1])))

            gbg_plus_plus.apply_DBGBC(eps_ratio=0.5,min_samples=2) # 3WC-GBNRS++ 中的生成算法，如果不要可以注释掉

            generated_balls = gbg_plus_plus.balls
            gbg_end_time = time.time() # 粒球生成结束
            # for ball in generated_balls:          # 检查
            #     print("中心：",ball.center)
            #     print("半径",ball.radius)
            #     print("label:",ball.label)
            #     print("purity:",ball.purity)
            #     print(ball.num_samples)
            #     break

            print(f"Generation time: {gbg_end_time - gbg_start_time:.4f}秒")
            print(f"Number of generated balls: {len(generated_balls)}")

            # 特征方法表示（实验中）
            X_train_gb = granular_ball_features(X_train, generated_balls)
            X_test_gb = granular_ball_features(X_test, generated_balls)


            print("\n[粒球特征性能]")
            # 粒球预测时间
            start_time_gb = time.time()
            gb_results = evaluate_model(X_train_gb, X_test_gb, y_train, y_test)
            end_time_gb = time.time()
            print(f"粒球时间: {end_time_gb - start_time_gb:.4f}秒")
            for model_name, metrics in gb_results.items():
                print(f"{model_name}:")
                print(f"  Accuracy: {metrics['Accuracy']:.4f}")
                print(f"  Precision: {metrics['Precision']:.4f}")
                print(f"  Recall: {metrics['Recall']:.4f}")
                print(f"  F1: {metrics['F1']:.4f}")

            print("\n[原始特征性能]")
            start_time_original = time.time()
            original_results = evaluate_model(X_train, X_test, y_train, y_test)
            end_time_original = time.time()
            print(f"原始特征时间: {end_time_original - start_time_original:.4f}秒")
            for model_name, metrics in original_results.items():
                print(f"{model_name}:")
                print(f"  Accuracy: {metrics['Accuracy']:.4f}")
                print(f"  Precision: {metrics['Precision']:.4f}")
                print(f"  Recall: {metrics['Recall']:.4f}")
                print(f"  F1: {metrics['F1']:.4f}")


            # 训练GBKNN++
            gbknn = GBKNNPlusPlus(purity_threshold=0.95)
            gbknn.fit(X_train, y_train)

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)

            gbknn_plus_features = GBKNNPlusPlus(purity_threshold=0.95)
            gbknn_plus_features.fit(X_train_gb, y_train)

            knn_plus_features = KNeighborsClassifier(n_neighbors=5)
            knn_plus_features.fit(X_train_gb, y_train)

            y_pred_GBKNN = gbknn.predict(X_test)
            y_pred_knn = knn.predict(X_test)
            y_pred_GBKNN_plus_features = gbknn_plus_features.predict(X_test_gb)
            y_pred_knn_plus_features = knn_plus_features.predict(X_test_gb)

            # 评估
            print(f"y_pred_GBKNN_Accuracy: {accuracy_score(y_test, y_pred_GBKNN):.4f}")
            print(f"y_pred_knn_Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
            print(f"y_pred_GBKNN_plus_features_Accuracy: {accuracy_score(y_test, y_pred_GBKNN_plus_features):.4f}")
            print(f"y_pred_knn_plus_features_Accuracy: {accuracy_score(y_test, y_pred_knn_plus_features):.4f}")


# 主程序部分
# if __name__ == "__main__":
#     # 数据准备（示例）
#     from sklearn.datasets import load_wine
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#     data = load_wine()
#
#     X = data.data
#     y = data.target
#     print(X.shape)
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     # 划分训练测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 获取Balls
#     gbg_plus_plus = GBGPlusPlus(purity_threshold=1.0)
#     gbg_plus_plus.fit(X_train, y_train)
#     generated_balls = gbg_plus_plus.balls
#     print(len(generated_balls))
#     # for ball in generated_balls:
#     #     print(ball.data)
#     #     print(ball.attributes)
#     #     print(ball.data_no_label)
#     #     break
#
#     print(generated_balls)
#     X_train_gb = granular_ball_features(X_train, generated_balls)
#     X_test_gb = granular_ball_features(X_test, generated_balls)
#
#     # 训练GBKNN++
#     gbknn = GBKNNPlusPlus(purity_threshold=1.0)
#     gbknn.fit(X_train, y_train)
#     # 传统knn
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(X_train, y_train)
#     # 特征表示的GBKNN++
#     gbknn_plus_features = GBKNNPlusPlus(purity_threshold=1.0)
#     gbknn_plus_features.fit(X_train_gb, y_train)
#     # 特征表示额传统knn
#     knn_plus_features = KNeighborsClassifier(n_neighbors=5)
#     knn_plus_features.fit(X_train_gb, y_train)
#     # 预测
#     y_pred_GBKNN = gbknn.predict(X_test)
#     y_pred_knn = knn.predict(X_test)
#     y_pred_GBKNN_plus_features = gbknn_plus_features.predict(X_test_gb)
#     y_pred_knn_plus_features = knn_plus_features.predict(X_test_gb)
#
#     # visualize_granular_balls(X_train, y_train, generated_balls, list(range(X_train.shape[1])))
#     # 评估
#     print("y_pred_GBKNN_Accuracy:", accuracy_score(y_test, y_pred_GBKNN))
#     print("y_pred_knn_Accuracy:", accuracy_score(y_test, y_pred_knn))
#     print("y_pred_GBKNN_plus_features_Accuracy:", accuracy_score(y_test, y_pred_GBKNN_plus_features))
#     print("y_pred_knn_plus_features_Accuracy:", accuracy_score(y_test, y_pred_knn_plus_features))