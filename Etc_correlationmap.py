
# coding: utf-8

# # 相関マップ
# ---
# 引数：raw_al45.csv/raw_al135.csvがあるディレクトリまでのパス
#
# ----
#
# 入力：raw_al45.csv/raw_al135.csv
#
# ----
#
# 出力：correlationmap.csv ボクセルごとに相関を算出したもの一覧
#
# ---
# ボクセルごとに相関を算出する．
# タスク1のZ-scoreには1を，タスク2のZ-scoreには-1をかけ，和を求める．値のものほど相関が大きいということになるため，昇順に並べ替えておく．

# In[1]:

print('############ Etc_correlationmap.py program excution ############')


# In[3]:

import numpy as np
import pandas as pd
import sys


# In[4]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../SpmActive/20181119tm/RawData/'


# In[5]:

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


# In[129]:

# ボクセル数
voxNum = len(al45) // 4

# 何ボクセル目かをカウント
counter = 0

# ボクセル名取得用
voxNames = []

# ボクセルごとの相関格納用
cormap = []

for voxNo in range(voxNum):

    voxName = 'Voxel' + str(voxNo + 1)

    # ボクセルのデータを取得
    al45Vox = al45.loc[voxName]
    al135Vox = al135.loc[voxName]

    # ボクセルごとに各タスクの総和を求める

    # 45度線は1をかけたものの総和
    # 135度線は-1をかけたものの総和
    # データフレームの値はObject型なのでfloat型に変換しないと掛け算や総和を求められない
    al45sum = sum(al45Vox.astype(float).sum())
    al135sum = sum((al135Vox.astype(float) * (-1)).sum())

    # 各タスクの総和を足すことで相関を求める
    alsum = al45sum + al135sum

    # 求めた相関を格納
    cormap = cormap + [alsum]

    print(voxName + '( ' + str(counter+1) + ' / ' + str(voxNum) + ' ) : ' + str(alsum))

    counter = counter + 1
    voxNames = voxNames + [voxName]


# In[141]:

# 相関一覧をデータフレーム化
cormap = pd.DataFrame(cormap)

# カラム名，インデックス名をつける
cormap.index = voxNames
cormap.columns = ['Correlation']


# In[145]:

# 相関の大き順に並べ替え
cormap_sort = cormap.sort_values('Correlation', ascending = False)


# In[146]:

# csv書き出し
PATH_cormap = PATH + 'correlationmap.csv'
cormap_sort.to_csv(PATH_cormap)


# In[ ]:
