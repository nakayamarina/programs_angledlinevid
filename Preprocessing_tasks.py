
# coding: utf-8

# # 実験から得られたfMRIデータの前処理
# ----
#
# 引数：Y01.csv, Y02.csv,... の入ったVexelディレクトリがあるディレクトリまでのパス
#
# ---
#
# 入力：Y01.csv, Y02.csv,...
#
# ---
#
# 出力：
# * RawData/raw_all.csv : すべてのボクセル45degrees,135degrees（rest時は除く）のZ-scoreをまとめたもの
# * RawData/raw_al45.csv : 45度の斜め線動画提示時のZ-scoreだけをまとめたもの
# * RawData/raw_al135.csv : 135度の斜め線動画提示時のZ-scoreだけをまとめたもの
# * RawData/Raw_image/voxel[ボクセル番号]-[試行数]_al45.png：45度の斜め線動画提示時の各ボクセルのデータをプロットしたもの
# * RawData/Raw_image/voxel[ボクセル番号]-[試行数]_al135.png：135度の斜め線動画提示時の各ボクセルのデータをプロットしたもの
#
# ----
#
#
# /VoxelディレクトリのY01.csv, Y02.csv, ... のデータには，選択してきた数ボクセルそれぞれのZ-score（賦活度合いみたいなもの）が記録されている．
#
# ここでは，全タスク，各タスクごとに分別した時系列データを得る．
#
#

# In[1]:

print('########## Preprocessing_tasks.py program excution ############')


# In[24]:

import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# コマンドライン引数で/Voxelディレクトリがあるディレクトリまでのパスを取得

# In[43]:

args = sys.argv
PATH_pre = args[1]

# # jupyter notebookのときはここで指定
# PATH_pre = '../Active/20181119mt/'

# /Voxelディレクトリまでのパス
PATH = PATH_pre + 'Voxel/'


# In[93]:

# plotするなら1，plotしないなら0
imgPlot = 1

# 試行数
runNum = 4

# restのスキャン数
restNum = 8

# 1タスクのスキャン数
taskNum = 88


# 後で出力するcsvファイルを保存するディレクトリ（RawData）、pngファイルを保存するディレクトリ（Raw_image）を作成

# In[94]:

# RawDataのディレクトリ名・パス
DIR_RAW = PATH + '../RawData'
PATH_RAW = DIR_RAW + '/'

# すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_RAW):
    os.mkdir(DIR_RAW)

# Raw_imageのディレクトリ名・パス
DIR_image = PATH_RAW + 'Raw_image'
PATH_image = DIR_image + '/'

# すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_image):
    os.mkdir(DIR_image)



# ## splitVoxRun関数
#
# 引数に全ボクセルのデータをまとめたデータフレームを受け取り，各ボクセルで試行ごとに分割結合する．

# In[95]:

def splitVoxRun(data):

    # 各試行ごとに分割，横結合（Voxel1-Run1, Voxel1-Run2, Voxel1-Run3, Voxel1-Run4, Voxel2-Run1, Voxel2-Run2...）

    # データ格納用
    vox_run_all = pd.DataFrame(index = [], columns = [])

    for i in range(len(data.columns)):

        # ボクセルで試行ごとに分割，reshapeを使って1列データを(試行数，1タスクのスキャン数)に
        vox_run = np.reshape(data.iloc[:, i], (runNum, taskNum))

        # 転置してデータフレーム化
        vox_run = pd.DataFrame(vox_run).T

        # 列名つける
        col_name = ['Voxel' + str(i + 1)] * runNum
        vox_run.columns = col_name

        # データ格納
        vox_run_all = pd.concat([vox_run_all, vox_run], axis = 1)

    return vox_run_all


# ## plotIMAGE関数
#

# In[153]:

def plotIMAGE(data, task):

    col_name = sorted(set(data.columns))

    # 何列目か
    i = 0

    # ボクセル（列）の数だけ繰り返す
    for vox_name in col_name:

        for j in range(runNum):

            # この後に出力するpngファイル名
            FILE_NAME = DIR_image + '/' + task + '-' + vox_name + '-Run' + str(j+1) + '.png'

            # データをplot
            plt.plot(data.iloc[:, i], label = 'fMRIdata')

            # グラフのタイトル
            graph_name = 'fMRIdata : ' + task + '-' + vox_name + '-Run' + str(j+1)
            plt.title(graph_name)
            plt.ylim([-5,5])
            plt.ylabel('Z-score')
            plt.xlabel('Time(scan)')

            # グラフの凡例
            plt.legend()

            # ファイル名をつけて保存，終了
            plt.savefig(FILE_NAME)
            plt.close()

            print(FILE_NAME)

            i = i + 1


# ## main関数

# * fMRIデータ読み込み
# * 全ボクセルデータ連結
# * 全ボクセルデータをcsvで書き出し

# In[154]:

if __name__ == '__main__':
    # /Voxelディレクトリ内のcsvファイルのパスを取得
    csv_file = PATH + '*.csv'
    files = []
    files = glob.glob(csv_file)
    files.sort()


# In[155]:

# 全ボクセルのデータをまとめる用
brain = pd.DataFrame(index = [], columns = [])

# 各ボクセルのZ-scoreが記録されたファイルの読み込み，結合
for i in range(len(files)):

    row_name = "Voxel" + str(i+1)
    data = pd.read_csv(files[i], names=(row_name,))

    brain = pd.concat([brain, data], axis = 1)


# In[156]:

# 各タスクごとのマスク作成
maskAl45 = (([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum)) * 4
maskAl135 = (([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum)) * 4


# In[157]:

# mask適用
dataAl45 = brain[maskAl45]
dataAl135 = brain[maskAl135]


# In[158]:

# splitVoxRun関数で各ボクセル，各試行で分割，csv書き出し
vox_run_al45 = splitVoxRun(dataAl45)
al45_file = PATH_RAW + 'raw_al45.csv'
vox_run_al45.to_csv(al45_file)

vox_run_al135 = splitVoxRun(dataAl135)
al135_file = PATH_RAW + 'raw_al135.csv'
vox_run_al135.to_csv(al135_file)


# In[159]:

if imgPlot == 1:

    plotIMAGE(vox_run_al45, 'angledlinevid-45degreea')
    plotIMAGE(vox_run_al135, 'angledlinevid-135degrees')


# In[ ]:
