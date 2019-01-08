
# coding: utf-8

# # SVMによるボクセルごとの学習と性能評価（単変量解析）
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
# 出力：ACCURACY[loo or k_cv]_VOXunivariate_SVM.csv ボクセルごとの識別性能評価結果一覧
#
# ----
#
# ボクセルごとに単変量解析を行う．
# k分割交差検証法により1グループをテストデータの，k-1グループを教師データとし，SVMを用いて学習，精度評価．
# ベクトル：各ボクセルにおけるある時刻のZ-score（1ベクトル）

# In[1]:

print('############ ML_SVM_VOXunivariate_kCV.py program excution ############')


# In[81]:

import numpy as np
import pandas as pd
import sys
from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[82]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../SpmActive/20181119tm/RawData/'

# 検証手法
col_name = 'leave-one-out'

# 検証手法
kCV = 10


# ## SVM_kCV関数
# 引く数としてデータをX，ラベルをyで受け取る．
# 交差検証法の一つk分割交差検証法で識別精度評価を行う．

# In[89]:

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



# # main関数

# In[90]:

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


# In[114]:

# ボクセル数
voxNum = len(al45) // 4

# 全ボクセルの識別率を格納するデータフレーム
voxAc = pd.DataFrame(index = range(voxNum), columns = [col_name])

counter = 0
csvcounter = 0
voxNames = []

for voxNo in range(voxNum):

    voxName = 'Voxel' + str(voxNo + 1)
    print(voxName + '( ' + str(counter+1) + ' / ' + str(voxNum) + ' )')

    # ボクセルのデータを取得
    al45Vox = al45.loc[voxName]
    al135Vox = al135.loc[voxName]

    # データセット作成
    al45Vox_vec = np.ravel(al45Vox)
    al135Vox_vec = np.ravel(al135Vox)

    data = np.r_[al45Vox_vec, al135Vox_vec]

    # データ数+1にするためにリシェイプ
    data = data.reshape(-1, 1)

    # ラベルを作成
    al45Vox_label = np.zeros(len(al45Vox_vec))
    al135Vox_label = np.ones(len(al135Vox_vec))

    labels = np.r_[al45Vox_label, al135Vox_label]

    # 学習と評価
    result_vox = SVM_kCV(data, labels)
    print(result_vox)

    # データフレームに格納
    voxAc.at[voxNo, :] = result_vox

    # 途中経過見る用
    # 何ボクセルで一度出力するか
    midNum = 1000

    if (counter % midNum == 0) and (counter != 0):

        PATH_test = PATH + 'ACMID' + str(csvcounter) + '[' + str(kCV) + 'cv]_VOXunivariate' + '_SVM.csv'
        print(PATH_test)
        MidVoxAc = voxAc.iloc[(csvcounter * midNum):((csvcounter + 1) * midNum), :]
        MidVoxAc.index = voxNames[(csvcounter * midNum):((csvcounter + 1) * midNum)]
        MidVoxAc.to_csv(PATH_test, index = True)

        csvcounter = csvcounter + 1

    counter = counter + 1
    voxNames = voxNames + [voxName]



# In[117]:

# 行名つける
voxAc.index = voxNames

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[' + str(kCV) + 'CV]_VOXunivariate' + '_SVM.csv'
voxAc.to_csv(PATH_RESULT, index = True)


# In[ ]:
