
# coding: utf-8

# # SVMによる相関の高いボクセルを用いた学習と性能評価（多変量解析）
# ----
#
# 引数：raw_al45.csv/raw_al135.csvがあるディレクトリまでのパス
#
# ----
#
# 入力：raw_al45.csv/raw_al135.csv
#
# ----
#
# 出力：ACCURACY[loo or k_cv]_CORmultivariate_SVM.csv ボクセルごとの識別性能評価結果一覧
#
# ----
#
# 相関の高いボクセルを用いて多変量解析を行う．
# k分割交差検証法により1グループをテストデータの，k-1グループを教師データとし，SVMを用いて学習，精度評価．
# ベクトル：各ボクセルにおけるある時刻のZ-score（ボクセル数ベクトル）

# In[1]:

print('############ ML_SVM_CORvariate_kCV.py program excution ############')


# In[2]:

import numpy as np
import pandas as pd
import sys
from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[159]:

args = sys.argv
PATH = args[1]

# # jupyter notebookのときはここで指定
# PATH = '../MaskBrodmann/20181119tm/10vox_RawData/'

# 検証手法
kCV = 10

# 試行数
runNum = 4

# 検証手法
col_name = str(kCV) + 'CV'


# ## SVM_kCV関数
# 引数としてデータをX，ラベルをyで受け取る．
# 交差検証法の一つk分割交差検証法で識別精度評価を行う．

# In[160]:

def SVM_kCV(X, y):

    # 線形SVMのインスタンスを生成
    model = svm.SVC(kernel = 'linear', C = 1)

    # k分割し，1グループをテストデータ，残りグループを教師データにして評価
    # すべてのグループに対して行う
    # 評価結果（識別率）を格納
    CVscore = cross_validation.cross_val_score(model, X, y, cv = kCV)

    # 評価結果（識別率）の平均を求める
    result = CVscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print('k = ' + str(kCV) + '：' + str(CVscore))

    return result


# ## Reshape関数
# 引数として整形したいデータをdataで受け取る．
# 各ボクセルが試行ごとに分割されているので1列にまとめ直して返す．

# In[161]:

def Reshape(data):

    # ボクセル数
    voxNum = len(data) // runNum

    # まとめ直したデータ格納用
    data_new = pd.DataFrame(index = [], columns = [])

    for counter in range(voxNum):

        # 各ボクセルのデータ取得
        vox = data.iloc[(counter * runNum):((counter + 1) * runNum), :]

        # ボクセル名取得
        voxName = list(set(vox.index))[0]
        print(voxName + '( ' + str(counter+1) + ' / ' + str(voxNum) + ' )')

        # １列データに変換する
        # reshapeを使うためにarray型に変換
        vox_arr = vox.as_matrix()

        # reshapeを使って1列データへ，データフレーム化
        vox_new = pd.DataFrame(vox_arr.reshape(-1))

        # カラム名をつける
        vox_new.columns = [voxName]

        data_new = pd.concat([data_new, vox_new], axis = 1)

    return data_new


# # main関数

# In[162]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_al45 = PATH + 'raw_al45.csv'
    PATH_al135 = PATH + 'raw_al135.csv'

    # csvファイル読み込み
    # headerは設定せず，転置後にset_index()する（header = 0にすると列名が変えられる）
    al45 = pd.read_csv(PATH_al45, header = None, index_col = 0).T
    al45.columns = range(0, len(al45.columns))
    al45 = al45.set_index(0)

    al135 = pd.read_csv(PATH_al135, header = None, index_col = 0).T
    al135.columns = range(0, len(al135.columns))
    al135 = al135.set_index(0)


# In[163]:

# データ整形
print('al45')
al45_new = Reshape(al45)

print('al135')
al135_new = Reshape(al135)


# In[164]:

# 各タスクのデータを結合
all_data = pd.concat([al45_new, al135_new], axis = 0)

# ベクトル化
X = all_data.as_matrix()


# In[165]:

# ラベル作成 al45 = 0, al135 = 1
label_al45 = np.zeros(len(al45_new))
label_al135 = np.ones(len(al135_new))

y = np.r_[label_al45, label_al135]


# In[166]:

# 学習と評価
result = SVM_kCV(X, y)
print(result)


# In[175]:

# データフレーム化する際のインデックス名作成
index_name = str(al45_new.shape[1]) + 'voxels'

# データフレーム化
result_df = pd.DataFrame({col_name:[result]}, index = [index_name])


# In[180]:

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[' + str(kCV) + 'CV]_CORmultivariate' + '_SVM.csv'
result_df.to_csv(PATH_RESULT)


# In[ ]:
