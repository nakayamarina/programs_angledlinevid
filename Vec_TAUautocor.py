
# coding: utf-8

# # 各ボクセルごとの時間遅れτを求める
# ### 自己相関関数が最初に極小値をとる時刻
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
# 出力：45度/135度の斜め線動画提示時の各ボクセルの時間遅れτをまとめたもの
#
# ---
#
# 時系列特性を得るために3次元空間に写像する．
# 時系列データにおいて，ある時刻tの値をx軸，t+τ（時間遅れ）の値をy軸，t+2*τの値をz軸に写像すると，
# 特徴的な軌道を描くとされている（カオス理論）．
# 時間遅れτの求め方はいくつかあるが，このプログラムでは時系列データ（各ボクセルのデータ）の自己相関関数が最初に極小値をとる時刻をτとする．

# In[1]:

print('########## TAUautocor.py program excution ############')


# In[2]:

import numpy as np
from scipy import signal
import sys
import pandas as pd


# コマンドライン引数でraw_al135.csv/raw_al45.csvがあるディレクトリまでのパスを取得

# In[6]:

args = sys.argv
PATH = args[1]

# # jupyter notebookのときはここで指定
# PATH = '../Active/20181119mt/RawData/'

# 読み込みたいファイルのパス
PATH_al45 = 'raw_al45.csv'
PATH_al135 = 'raw_al135.csv'


# ## autocor関数
# 引数としてmain関数で読み込んだデータをdataで受け取る．
# rock，scissor，paperの各ボクセルごとの自己相関関数が最初に極小値をとる時刻を調べる --> csvファイルで書き出し

# In[4]:

def autocor(data):

    # 求めた値を入れる
    TAUs = []

    # ボクセル（列）の数だけ繰り返す
    for i in range(len(data.columns)):

        # i番目のボクセルデータ抽出
        voxel = data.iloc[:, i]

        # 自己相関関数
        x = np.correlate(voxel, voxel, mode = 'full')

        # 極小値のインデックス一覧
        first_min = signal.argrelmin(x)

        # 「最初に極小値をとるときの値」なので最初の値をTAUsに追加
        TAUs.append(first_min[0][0])

    print(TAUs)

    return TAUs


# ## main関数
#
# * tap_raw.csv/rest_raw.csv読み込み
# * autcor関数呼び出し

# In[12]:

if __name__ == '__main__':

    # csvファイル読み込み
    al45 = pd.read_csv(PATH + PATH_al45, header = 0, index_col = 0)
    al135 = pd.read_csv(PATH + PATH_al135, header = 0, index_col = 0)



# In[13]:

tau_al45 = autocor(al45)


# In[14]:

tau_al135 = autocor(al135)


# In[15]:

# RestとTappingの各ボクセルごとの時間遅れTAUを整形
TAUs = pd.DataFrame({'1':tau_al45, '2':tau_al135})
TAUs.columns = ['al45', 'al135']


# In[16]:

# csv書き出し
PATH_TAU = PATH + 'TAUautocor.csv'
TAUs.to_csv(PATH_TAU, index = False)


# In[ ]:
