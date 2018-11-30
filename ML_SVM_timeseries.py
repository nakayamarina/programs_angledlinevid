
# coding: utf-8

# # SVMによる学習と識別性能評価（時系列解析）

# ---
#
# 引数：raw_al45.csv/raw_al135.csvがあるディレクトリまでのパス
#
# ---
#
# 入力：raw_al45.csv/raw_al135.csv
#
# ---
#
# 出力：ACCURACY[loo or k_cv]_timeseries_SVM.csv　識別性能評価結果一覧
#
# ---
#
# 生データをSVMを用いて学習し，交差検証法（k-分割交差検証，leave-one-out交差検証）を用いて識別性能評価を行う．
# ベクトル：各ボクセルの時系列データ
#
#

# In[1]:

print('############# ML_SVM_timeseries.py program excution ##############')


# In[2]:

import numpy as np
import pandas as pd
import sys

from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split


# コマンドライン引数でraw_al45.csv/raw_al135.csvがあるディレクトリまでのパスを取得

# In[8]:

args = sys.argv
PATH = args[1]

# # jupyter notebookのときはここで指定
# PATH = '../Active/20181119mt/RawData/'

# 検証手法
col_name = 'leave-one-out'


# ## SVM_LOO関数
# 引数として教師データをX，ラベルをyで受け取る．
# 交差検証法の一つleave-one-out交差検証で識別精度評価を行う．
#
# * (1個をテストデータ，残りを教師データにして学習・評価) * すべてのデータ個
# * 得られたすべてのデータ個の評価結果（識別率）の平均を求めてパーセントに直す
# * 評価結果（識別率）をmain関数に返す

# In[4]:

def SVM_LOO(X, y):

    LOOscore = np.zeros(len(X))

    # 1個をテストデータ，残りを教師データにして学習・評価
    # すべてのデータに対して行う
    for i in range(len(X)):

        print('------ ' + str(i + 1) + ' / ' + str(len(X)) + '回 -----')

        # テストデータ
        X_test = X[i].reshape(1, -1)
        y_test = y[i].reshape(1, -1)

        # テストデータとして使用するデータを除いた教師データを作成
        X_train = np.delete(X, i, 0)
        y_train = np.delete(y, i, 0)

        # 線形SVMのインスタンスを生成
        model = svm.SVC(kernel = 'linear', C = 1)

        # モデルの学習
        model.fit(X_train, y_train)

        # 評価結果（識別率）を格納
        LOOscore[i] = model.score(X_test, y_test)


    # 評価結果（識別率）の平均を求める
    result = LOOscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print(str(LOOscore) + '\n')

    return result


# ## SVM_kCV関数
# 引数とし教師データをX，ラベルをyで受け取る．
# 交差検証法の一つk-分割交差検証で識別精度評価を行う．
#
# * 学習
# * (k分割し，1グループをテストデータ，残りグループを教師データにして評価) * k
# * 得られたk個の評価結果（識別率）の平均を求めてパーセントに直す
# * 評価結果（識別率）をmain関数に返す

# In[5]:

def SVM_kCV(X, y):

    # 線形SVMのインスタンスを生成
    model = svm.SVC(kernel = 'linear', C = 1)

    # k分割し，1グループをテストデータ，残りグループを教師データにして評価
    # すべてのグループに対して行う
    # 評価結果（識別率）を格納
    CVscore = cross_validation.cross_val_score(model, X, y, cv = cv_k)

    # 評価結果（識別率）の平均を求める
    result = CVscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print('k = ' + str(cv_k) + '：' + str(CVscore))

    return result



# ## TrainingData関数
# 引数として読み込みたいタスクごとのデータをdata1/data2で受け取る．
# * 機械学習にかけれるようにデータのベクトル化とラベルを作成
# * ベクトル化したデータとラベルをmain関数に返す

# In[17]:

def TrainingData(data1, data2):

    # 各タスクのデータを縦結合
    all_data = pd.concat([data1, data2], axis = 0)

    # ベクトル化
    X = all_data.as_matrix()

    # ラベル作成 data1 = 0, data2 = 1
    label_data1 = np.zeros(len(data1.index))
    label_data2 = np.ones(len(data2.index))

    y = np.r_[label_data1, label_data2]


    return X, y




# ## main関数

# In[18]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_al45 = PATH + 'raw_al45.csv'
    PATH_al135 = PATH + 'raw_al135.csv'

    # csvファイル読み込み，時系列解析のため行列を入れ替える
    al45 = pd.read_csv(PATH_al45, header = 0, index_col = 0)
    al45 = al45.T

    al135 = pd.read_csv(PATH_al135, header = 0, index_col = 0)
    al135 = al135.T

    # データとラベルの準備
    data, labels = TrainingData(al45, al135)



# In[15]:


# 学習とk-分割交差検証

print('leave-one-out cross-validation')

result_loo = SVM_LOO(data, labels)

# データフレーム化
result_loo = pd.DataFrame({col_name:[result_loo]}, index = ['SPM + SVM'])
print(result_loo)


# In[16]:

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[loo]_timeseries_SVM.csv'
result_loo.to_csv(PATH_RESULT, index = True)


# In[ ]:
