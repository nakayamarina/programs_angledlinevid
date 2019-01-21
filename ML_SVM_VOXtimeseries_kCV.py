
# coding: utf-8

# # SVMによるボクセルごとの学習と性能評価（時系列解析）
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
# 出力：ACCURACY[loo or k_cv]_VOXtimeseries[1時系列あたりのスキャン数]_SVM.csv ボクセルごとの識別性能評価結果一覧
#
# ----
# ボクセルごとにあるデータにおけるNスキャン分の時系列データをテストデータ，テストデータを除いた残りのデータにおけるNスキャン分の時系列データ1ずつずらしながらを学習データとして取得し，SVMを用いて学習，交差検証法を用いて識別性能評価を行う．
# ベクトル：各ボクセルにおけるデータにおけるNスキャン分の時系列データ

# In[41]:

print('############ ML_SVM_VOXtimeseries_kCV.py program excution ############')


# In[31]:

import numpy as np
import pandas as pd
import sys

from sklearn import svm


# In[32]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../SpmActive/20181119tm/RawData/'

# 1時系列あたりのスキャン数
N = 30

kCV = 10

# 検証手法
col_name = str(kCV) + 'CV'

# 試行数
runNum = 4


# In[33]:

# 機械学習に用いるデータ詳細を格納
data_info = pd.DataFrame(index = [], columns = [])

# 機械学習に用いるテストデータ，教師データ詳細を格納
vec_info = pd.DataFrame(index = [], columns = [])


# ## TsShift関数
# 引数としてあるボクセルにおける全試行分のデータをdata，タスクを見分けるための番号をlabelに受け取る．
# 各試行で1ずつずらしながらNスキャン分の時系列データを取得する．全試行の時系列データをまとめて返す．

# In[34]:

def TsShift(data, label):

    # 1ボクセルあたりで取得する時系列データ格納用データフレーム
    ts_all = pd.DataFrame(index = [], columns = [])

    for runNo in range(runNum):

        # 1試行あたりので得られる全データ
        runData = data.iloc[runNo, :]

        # 時系列として扱う区間の始まり
        ts_fst = 0

        # 時系列として扱う区間の終わり
        ts_end = N

        # 1ずつずらしながら時系列データを取得，結合
        while ts_end <= len(runData):

            # 時系列として扱う区間の始まり，区間の終わり，その区間（Nスキャン分）の時系列データの順でリスト化
            ts = [label] + [runNo + 1] + [ts_fst + 1] + [ts_end] + list(runData[ts_fst:ts_end])

            # データフレーム化，結合
            ts = pd.DataFrame(ts).T

            ts_all = pd.concat([ts_all, ts])

            # 1ずつずらす
            ts_fst = ts_fst + 1
            ts_end = ts_end + 1


    return ts_all


# ## SVM_kCV関数
# 学習と評価に用いるデータをdata，ボクセル名をkeyで受け取り．
# データからテストデータ，テストデータラベル，教師データ，教師データラベルを生成．
# テストデータはdataにおける全時系列データ，教師データは条件に当てはまる時系列データであり，
# 10分割交差検証法を用いて識別率を算出，平均を計算し，main関数へ返す．

# In[35]:

def SVM_kCV(data, key):

    scores = np.zeros(kCV)

    testNum = len(data)//kCV

    vec_info_sub = pd.DataFrame(index = [], columns = [])

    for i in range(kCV):

        print("--------" + str(i + 1) + " / " + str(kCV) + "---------")

        # kCV回目の評価では端数分のデータを入れる
        if i == (kCV-1):

            # テストデータの情報（タスク，試行数，時系列データの始まり，終わり）を取得
            test_labels = data.iloc[(i * testNum):(len(data)), 0]
            test_Flabel = data.iloc[i * testNum, 0]
            test_Elabel = data.iloc[(len(data) - 1), 0]

            test_Frun = data.iloc[i * testNum, 1]
            test_Erun = data.iloc[(len(data) - 1), 1]

            test_fst = data.iloc[i * testNum, 2]
            test_end = data.iloc[(len(data) - 1), 3]

            ###### テストデータ

            # 時系列データのみを抽出
            X_test = data.iloc[(i * testNum):(len(data)), 4:len(data.columns)]


        else:


            # テストデータの情報（タスク，試行数，時系列データの始まり，終わり）を取得
            test_labels = data.iloc[(i * testNum):((i + 1) * testNum), 0]
            test_Flabel = data.iloc[i * testNum, 0]
            test_Elabel = data.iloc[((i + 1) * testNum), 0]

            test_Frun = data.iloc[i * testNum, 1]
            test_Erun = data.iloc[(((i + 1) * testNum) - 1), 1]

            test_fst = data.iloc[i * testNum, 2]
            test_end = data.iloc[(((i + 1) * testNum) - 1), 3]

            ###### テストデータ

            # 時系列データのみを抽出
            X_test = data.iloc[(i * testNum):((i + 1) * testNum), 4:len(data.columns)]

        # ベクトル化
        X_test = X_test.as_matrix()
        # ラベルを作成
        y_test = np.array(list(test_labels))


        ###### 教師データ

        # テストデータではない，テストデータに含まれるZ-scoreを含まないものを取得

        if test_Flabel == test_Elabel:

            test_label = test_Flabel

            if test_Frun == test_Erun:

                test_run = test_Frun

                traindata = data[(data['label'] != test_label) | (data['run'] != test_run) | (data['fst'] > test_end) | (data['end'] < test_fst)]

            else:

                traindata = data[(data['label'] != test_label) | ( (data['run'] != test_Frun) & (data['run'] != test_Erun)) | ((data['run'] == test_Frun ) & (data['end'] < test_fst)) | ((data['run'] == test_Erun) & (data['fst'] > test_end))]

        else:

            traindata = data[((data['label'] == test_Flabel) & (data['run'] != test_Frun)) | ((data['label'] == test_Elabel) & (data['run'] != test_Erun))
                            | ((data['label'] == test_Flabel) & (data['run'] == test_Frun) & (data['end'] < test_fst))
                            | ((data['label'] == test_Elabel) & (data['run'] == test_Erun) & (data['fst'] > test_end))]


        ####### 機械学習に用いるデータ詳細 #######

        if key == 'Voxel1':

            # 全てのデータのラベル・試行数・Scan始まり番号情報
            all_lrf = data.iloc[:, 0:4]

            # ある1グループをテストデータとした時の教師データのラベル・試行数・Scan始まり番号情報
            cv_lrf = traindata.iloc[:, 0:4]

            # 教師データであることを示す'train'を結合
            cv_one = pd.DataFrame(['train'] * len(cv_lrf))
            cv_one.index = cv_lrf.index
            cv_lrf = pd.concat([cv_lrf, cv_one], axis = 1)

            # mergeすることで教師データは全データのうちのどこなのかを一覧化する
            lrf = pd.merge(all_lrf, cv_lrf, on=['label', 'run', 'fst', 'end'], how = 'left')

            # ラベル・試行数・Scan始まり番号情報を結合して，インデックスとする
            train_test = pd.DataFrame(lrf.iloc[:, len(lrf.columns)-1])
            lrf_list = list(lrf['label'].astype(str) + '-' + lrf['run'].astype(str) + '-' + lrf['fst'].astype(str) + '-' + lrf['end'].astype(str))
            train_test.index = lrf_list

            # ボクセル番号，テストデータの最初のデータの情報と最後のデータの情報（ラベル-試行数-最初or最後のScan数），
            # テストデータor教師データの一覧の順に並べたデータフレーム作成（横長）
            train_test = train_test.T

            test_info = pd.DataFrame([str(test_Flabel) + '-' + str(test_Frun) + '-' + str(test_fst)] + [str(test_Elabel) + '-' + str(test_Erun) + '-' + str(test_end)]).T
            test_info.columns = ['Test-l-r-fst', 'Test-l-r-end']

            vox_df = pd.DataFrame([key])
            vox_df.columns = ['BAvoxelNum']

            vox_df_test_info = pd.concat([vox_df, test_info], axis = 1)
            vox_df_test_info_train_test = pd.concat([vox_df_test_info, train_test], axis = 1)

            vec_info_sub = pd.concat([vec_info_sub, vox_df_test_info_train_test])

        ###################################


        # 時系列データのみを抽出
        X_train = traindata.iloc[:, 4:len(data.columns)]

        # ベクトル化
        X_train = X_train.as_matrix()

        # ラベルを作成
        y_train = np.array(list(traindata['label']))

        # 線形SVMのインスタンスを生成
        model = svm.SVC(kernel = 'linear', C=1)

        # モデルの学習
        model.fit(X_train, y_train)

        # 評価結果を格納
        scores[i] = model.score(X_test, y_test)

    # 評価結果の平均を求める
    result = scores.mean()


    # パーセント表記へ
    result = round(result * 100, 1)

    print(str(scores) + '\n')

    ####### 機械学習に用いるデータ詳細 #######

    if key == 'Voxel1':

        # vec_infoがグローバル変数であることを示してmerge
        global vec_info
        vec_info = pd.concat([vec_info, vec_info_sub])

        # 詳細csv書き出し

        details = pd.merge(data_info, vec_info, on=['BAvoxelNum'], how='right')

        PATH_dt = PATH + 'DETAILS[' + str(kCV) + 'CV]_VOXtimeseriesDetails' + str(N) + '_SVM.csv'
        details.to_csv(PATH_dt)

        print(PATH_dt)

    ###################################


    return result


# # main関数

# In[36]:

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


# In[37]:

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

    # ボクセルにおける時系列データを取得
    al45VoxTs = TsShift(al45Vox, 0)
    al135VoxTs = TsShift(al135Vox, 1)

    # 全タスクを縦結合
    VoxTs = pd.concat([al45VoxTs, al135VoxTs])

    # 0-3列目は条件判定用の要素，要素名をつけておく
    col_names = list(VoxTs.columns)
    col_names[0:4] = ['label', 'run', 'fst', 'end']
    VoxTs.columns = col_names

    VoxTs.index = range(0,len(VoxTs))

    ####### 機械学習に用いるデータ詳細 #######

    # data_infoに情報を格納する
    Dinfo = pd.DataFrame({'TimeSeries(Scan)':[N], 'kCV':[kCV], 'BAvoxelNum':[voxName], 'Data':[len(VoxTs)]})
    data_info = pd.concat([data_info, Dinfo])

    ######################################

    # 学習と評価
    result_vox = SVM_kCV(VoxTs, voxName)

    print(result_vox)

    # データフレームに格納
    voxAc.at[voxNo, :] = result_vox

    # 途中経過見る用
    # 何ボクセルで一度出力するか
    midNum = 1000

    if (counter % midNum == 0) and (counter != 0):

        PATH_test = PATH + 'ACMID' + str(csvcounter) + '[' + str(kCV) + 'cv]_VOXtimeseries' + str(N) +'_SVM.csv'
        print(PATH_test)
        MidVoxAc = voxAc.iloc[(csvcounter * midNum):((csvcounter + 1) * midNum), :]
        MidVoxAc.index = voxNames[(csvcounter * midNum):((csvcounter + 1) * midNum)]
        MidVoxAc.to_csv(PATH_test, index = True)

        csvcounter = csvcounter + 1

    counter = counter + 1
    voxNames = voxNames + [voxName]




# In[38]:

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[' + str(kCV) + 'CV]_VOXtimeseries' + str(N) +'_SVM.csv'
voxAc.to_csv(PATH_RESULT, index = True)

# 行名つける
voxAc.index = voxNames

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[' + str(kCV) + 'CV]_VOXtimeseries' + str(N) +'_SVM.csv'
voxAc.to_csv(PATH_RESULT, index = True)



# In[ ]:
