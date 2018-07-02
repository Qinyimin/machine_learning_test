import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn import metrics
import time

# 读取mnist数据集
mnist = sio.loadmat('mnist.mat')
dataMat = mnist['data']
labelMat = mnist['label']

# 读取5个划分的索引
index = sio.loadmat('index.mat')
train_index = index['train_index']
test_index = index['test_index']


def svm_baseline(i):
    start_time = time.time()
    # 设定分类器参数
    clf = svm.SVC(C=100, kernel='rbf', gamma=0.03,decision_function_shape='ovo')
    # 使用第i个训练集的数据和样本进行模型训练
    clf.fit(dataMat[train_index[i]],labelMat[0][train_index[i]])
    # 第i个测试集的预测结果
    predict_result=clf.predict(dataMat[test_index[i]])
    # print(predict_result)
    predictions = [int(a) for a in predict_result]
    # print(predictions)
    # num_correct = sum(int(a == y) for a, y in zip(predictions, labelMat[0][test_index[i]]))
    # print("%s of %s test values are correct." % (num_correct, len(labelMat[0][test_index[i]])))
    score = clf.score(dataMat[test_index[i]],labelMat[0][test_index[i]])
    print("Score: {:.6f}.".format(score))
    print("Error rate is {:.6f}.".format((1 - score)))
    f_measure=metrics.f1_score(labelMat[0][test_index[i]],predict_result,average='micro')
    print("F-measure: {:.6f}.".format(f_measure))
    end_time = time.time()
    print("Testing test set {} spent {:.2f}s.".format(i+1,end_time-start_time))
    print("---------------------------------------------------------")
    return score


if __name__ == "__main__":
    start_time=time.time()
    # score_list 用于保存5次测试的精度
    score_list=[]
    # 分别对5个训练集和测试集进行训练和测试
    for i in range(0,5):
        score=svm_baseline(i)
        score_list.append(score)

    # print(score_list)
    end_time=time.time()
    avgAccuracy = np.mean(score_list)
    print("Average accuracy is: {:.6f}.".format(avgAccuracy))
    avgStd=np.std(score_list)
    print("Standard deviation is: {:.6f}.".format(avgStd))
    print("Testing all 5 test sets spent {:.2f}s.".format(end_time - start_time))
    print("---------------------------------------------------------")

