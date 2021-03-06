
# coding: utf-8

# # NiftiファイルからZ-scoreを自動抽出
# ---
#
# 引数：4D化した.niiファイルがあるフォルダまでのパス / 書き出すcsvの保存先 / mask名
#
# ---
#
# 入力：4D化した.niiファイルがあるフォルダまでのパス / 書き出すcsvの保存先 / mask名（chunks_list.csv, targets_list.csv）
#
# ---
#
# 出力：all_raw.csv / raw_al45.csv / raw_al135.csv（chunks_list.csv, targets_list.csv）
#
# ---
#
# 注意：このプログラムはUbuntuでやったほうがいいかも（抽出する部位が多いほどデータが重くなってメモリがたりなくなる！）
#
#
# 注意：このプログラムを実行する前に以下の作業をすること
#
# 3D脳画像データ（前処理済みの.niiファイル）を4Dに変換
#
# 1. SPM起動
# 2. Batchを開く
# 3. 画面左上SPM → Util → 3D to 4D File Conversion
# 3. 3D Volumesから4Dに変換したいファイル('swr'のついたniftiファイル全て)を選択
# 4. Output Filenameでファイル名を指定
# 5. 再生ボタンを押す
#
# Motor mask（Z-score化したいボクセルの位置）の作成
# ※ wfu_pickatlasはmatlabサーバーで実行（Macだと.nii形式で保存できなかった）
# ※ 中山メモ：ubuntuでshareで作業するにはmatlabまで移動してshareを開く，基本的にsudoを使って作業（matlabもsudoで起動）
#
# 1. wfu_pickatlasフォルダをダウンロード（ネットから or Shareかnakayamaフォルダのどっかにある）
# 2. MATLABでwfu_pickatlasフォルダのパスを通す
# 3. MATLABコマンドライン上で > wfu_pickatlas で起動
# 4. 画面左からマスクしたい部分を選択（全脳の場合は TD Hemispheres を全選択）
# 5. SAVE MASKからマスク画像をmaskとして保存（名前に'-'とか入れない）
# 6. SPM → fMRI起動
# 7. MenuからNormalize(Est & Wri)を選択
# 8. Bath Editorが立ち上がったらDataをダブルクリック
# 9. Image to Alignには'swr'がファイル名についているniftiファイルを適当に1つ選択
# 10. Image to Writeには先ほど作成したmaskファイルを選択
# 11. 再生ボタンを押す → wmask.niiができる
# 12. Batch Editorに戻りRealign(Reslice)を選択
# 13. Imagesをダブルクリック
# 14. 'swr'がファイル名についているniftiファイルを適当に一つ，先ほどできたwmask.niiファイルの順で選択
# 15. 再生ボタンを押す → rwmask.niiができる
#

# In[1]:

from mvpa2.suite import *
from mvpa2.datasets.mri import fmri_dataset
import os
import os.path
from os.path import join as pathjoin
from pprint import pprint
# from nifti import NiftiImage
import glob
import numpy as np
import pandas as pd
import sys
import pickle
# import dill
import csv

import nibabel as nib


# コマンドライン引数で.niiファイルがあるディレクトリまでのパスを取得．

# In[2]:

# !!!!!!!!!! scan_numらへん，dataset，zscoreのパラメータ書き換え


args = sys.argv
PATH = args[1]
PATH_save = args[2]
MASK_name = args[3]

# jupyter notebookのときはここで指定
# PATH = '../../Data_mri/angledlinevid-1fe/20181119tm/'
# PATH_save = '../MaskBrodmann/20181227rn/'
# MASK_name = 'rwmaskBA.nii'

PATH_mask = PATH + MASK_name

# 前処理済みならswrがついた.niiファイルを選択
PATH_nii = PATH + '4D.nii'



# In[3]:

# RawDataのディレクトリ名・パス
DIR_RAW = PATH_save + 'RawData'
PATH_RAW = DIR_RAW + '/'

# すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_RAW):
    os.mkdir(DIR_RAW)


# In[5]:

headcoil = PATH.split('/')[5]


# 総スキャン数
scan_num = 200

# restのスキャン数
restNum = 8

# 1タスクのスキャン数
taskNum = 88

# 試行数
runNum = 4


# ## splitTaks関数
#
# 全てのZ-scoreをまとめたものdataで受け取り，Rock時とscissor時とpaper時のデータを分けてcsvファイルで書き出し

# In[10]:

def splitTask(brain):

    # 各タスクごとのマスク作成
    maskAl45 = ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum)

    maskAl135 = ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum) + ([False] * restNum) + ([True] * taskNum) + ([False] * restNum) + ([False] * taskNum) + ([False] * restNum)

    # mask適用
    dataAl45 = brain[maskAl45]

    dataAl135 = brain[maskAl135]

    # splitVoxRun関数で各ボクセル，各試行で分割，csv書き出し
    vox_run_al45 = splitVoxRun(dataAl45)
    al45_file = PATH_RAW + 'raw_al45.csv'
    vox_run_al45.to_csv(al45_file)
    print(al45_file)

    vox_run_al135 = splitVoxRun(dataAl135)
    al135_file = PATH_RAW + 'raw_al135.csv'
    vox_run_al135.to_csv(al135_file)
    print(al135_file)



# ## splitVoxRun関数
#
# 引数に全ボクセルのデータをまとめたデータフレームを受け取り，各ボクセルで試行ごとに分割結合する．

# In[6]:

def splitVoxRun(data):

    # 各試行ごとに分割，横結合（Voxel1-Run1, Voxel1-Run2, Voxel1-Run3, Voxel1-Run4, Voxel2-Run1, Voxel2-Run2...）

    # データ格納用
    vox_run_all = pd.DataFrame(index = [], columns = [])

    for i in range(len(data.columns)):

        print('----- Voxel' + str(i + 1) + ' -----')

        # ボクセルで試行ごとに分割，reshapeを使って1列データを(試行数，1タスクのスキャン数)に
        vox_run = np.reshape(list(data.iloc[:, i]), (runNum, taskNum))

        # 転置してデータフレーム化
        vox_run = pd.DataFrame(vox_run).T

        # 列名つける
        col_name = ['Voxel' + str(i + 1)] * runNum
        vox_run.columns = col_name

        # データ格納
        vox_run_all = pd.concat([vox_run_all, vox_run], axis = 1)

    return vox_run_all



# # main関数

# * fMRIデータ（niftiファイル）読み込み
# * 全ボクセルデータZ-score化，抽出，書きだし

# In[37]:

if __name__ == '__main__':

    # 4D化した.niiファイル名リストを作成

    nifti = [PATH_nii]


    # In[8]:

    # 教師データの作成（この作業なしにやる方法がわからないので，必要のない作業ではあるがやる）

    task_p1 = ['-1'] * restNum
    task_p2 = ['-1'] * restNum

    task45 = ['0'] * taskNum
    task135 = ['1'] * taskNum
    taskRest = ['-1'] * restNum

    # pattern1:Rest -> 45 -> Rest -> 135 -> Rest
    task_p1.extend(task45)
    task_p1.extend(taskRest)
    task_p1.extend(task135)
    task_p1.extend(taskRest)

    # pattern2:Rest -> 135 -> Rest -> 45 -> Rest
    task_p2.extend(task135)
    task_p2.extend(taskRest)
    task_p2.extend(task45)
    task_p2.extend(taskRest)

    task = task_p2 + task_p1 + task_p2 + task_p2

    target = pd.DataFrame(task)

    PATH_target = PATH + 'targets_list.csv'
    target.to_csv(PATH_target, index = False, header = None)
    print('target')

    targets_list = []

    targets_file = open(PATH_target, 'rU')
    dataReader = csv.reader(targets_file)

    for row in dataReader:
        targets_list.append(row[0])


    # In[44]:

    # チャンク（試行数リスト？）の作成（この作業なしにやる方法がわからないので，必要のない作業ではあるがやる）

    chunk = ['1'] * scan_num
    chunk0 = ['2'] * scan_num
    chunk1 = ['3'] * scan_num
    chunk2 = ['4'] * scan_num

    chunk.extend(chunk0)
    chunk.extend(chunk1)
    chunk.extend(chunk2)

    chunks = pd.DataFrame(chunk)

    PATH_chunk = PATH + 'chunks_list.csv'
    chunks.to_csv(PATH_chunk, index = False, header = None)
    print('chunks')

    chunks_list = []

    chunks = open(PATH_chunk, 'rU')

    for x in chunks:
        chunks_list.append(x.rstrip('\r\n'))

    chunks.close()


    # In[52]:

    # データセットの整形

    dataset = fmri_dataset(nifti, targets=targets_list, chunks=chunks_list, mask=PATH_mask ,sprefix='voxel', tprefix='time', add_fa=None)

    print('dataset ready')

    poly_detrend(dataset, polyord=1, chunks_attr='chunks')

    dataset = dataset[np.array([l in ['-1', '0', '1']
                               for l in dataset.targets], dtype='bool')]


    # In[53]:

    # Z-score抽出 → データフレーム化
    zscore(dataset, chunks_attr='chunks', param_est=('targets', ['0', '1']), dtype='float32')
    print('z-score')

    VoxelData = pd.DataFrame(np.array(dataset[:,:]))
    print(VoxelData .shape)


    # In[54]:

    voxName = []

    for i in range(len(VoxelData.columns)):

        name = 'Voxel' + str(i+1)
        print(name)

        voxName.append(name)

    VoxelData.columns = voxName


    # In[24]:

    # 全データのcevファイル書き出し
    # PATH_all = PATH_RAW + 'all_raw.csv'
    # VoxelData.to_csv(PATH_all, index = False)
    # print(PATH_all)


    # In[60]:

    splitTask(VoxelData)


    # In[ ]:




    # In[19]:

    # もしデータセットの整形でサイズエラーが出た時は以下で.niiファイル読み込みしてサイズを確認する
    # nii0 = nib.load('../../Data_mri/tappingState-2fe/20181029rn/64ch/wmaskAll.nii')
    # img0 = nii0.get_data()
    # img0.shape


    # In[ ]:
