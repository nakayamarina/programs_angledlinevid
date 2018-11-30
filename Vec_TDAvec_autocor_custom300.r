# TDAによる特徴抽出とベクトル化（穴の種類やきざみ時間変更可）

# ---

# 引数：raw_al45.csv/raw_al135.csv, 時間遅れτのcsvファイルがあるディレクトリまでのパス

# ---

# 入力：raw_al45.csv/raw_al135.csv, 時間遅れτのcsvファイル(TAUautocre.csv)

# ---

# 出力：
# * TDAvec_autocor_al45_(パラメータ).csv：前処理をしたal45時のデータの特徴抽出を行ったもの
# * TDAvec_autocor_al135_(パラメータ).csv：前処理をしたal135時のデータの特徴抽出を行ったもの
#
# 場合に応じて
# * TDAvec_autocor_attractor/angledlinevid-45degrees-Voxel[ボクセル番号]-Run[試行数].png
# * TDAvec_autocor_attractor/angledlinevid-45degrees-Voxel[ボクセル番号]-Run[試行数].png
# * TDAvec_autocor_barcode/angledlinevid-45degrees-Voxel[ボクセル番号]-Run[試行数].png
# * TDAvec_autocor_barcode/angledlinevid-45degrees-Voxel[ボクセル番号]-Run[試行数].png


# ---

# 前処理をした

# * 45度の斜め線動画提示時のデータ
# * 135度の斜め線動画提示時のデータ

# の特徴抽出をTDAを用いて行う．

# (1) 3次元空間への写像
# 45度の斜め線動画提示時と135度の斜め線動画提示時の各ボクセルの時間遅れτを用いて
# 各ボクセルの時系列データにおいてある時刻tの値，t+τの値，t+2*τの値で3次元データとし，アトラクタ図形を得る

# (2) TDA適用
# 3次元データに対してTDAのPersistent Homology適用しバーコードダイアグラムを得る

# (3) ベクトル化
# 穴の数をkizamiNumber(TDAvec関数の変数)回数えることでベクトル化する

if (!require(package = "TDA")) {
  install.packages(pkgs = "TDA")
}

if (!require(package = "scatterplot3d")) {
  install.packages(pkgs = "scatterplot3d")
}

library(TDA)
library(scatterplot3d)

print('################ TDAvec_autocor.r excution ###################')

# コマンドライン引数でraw_al135.csv/raw_al45.csv, TAUautocor.csvがあるディレクトリまでのパスを取得

#PATH <- '../State-2fe_Active/20181029rn/64ch/RawData/'
PATH = commandArgs(trailingOnly=TRUE)[1]

# 読み込みたいファイルのパス
PATH_al45 <- paste(PATH, 'raw_al45.csv', sep = "")
PATH_al135 <- paste(PATH, 'raw_al135.csv', sep = "")
PATH_tau <- paste(PATH, 'TAUautocor.csv', sep = "")

# 試行数
runNum <- 4

# TDAvec関数で使うripsDiagのmaxsxaleの値設定
ms<- 3

# BettiNumberCount関数で使う穴を数える回数
kizamiNum <- 300
parameters <- paste("012dim", kizamiNum, sep = "")



# アトラクタ図，バーコードを出力する(1) or しない (0)
# 出力する場合の保存ディレクトリ名
atrct_output <- 0
barcode_output <- 0
atrctName <- 'TDAvec_autocor_attractor'
barcodeName <- 'TDAvec_autocor_barcode'


# main関数

# * raw_al135.csv/raw_reset.csv読み込み
# * TDAvec_autocor_al135.csv/TDAvec_autocor_al45.csv書き出し

main <- function(){

  # csvファイルの読み込み
  al45 <- read.csv(PATH_al45, row.names=1)
  al135 <- read.csv(PATH_al135, row.names=1)
  taus <- read.csv(PATH_tau)

  # ベクトル化したデータを格納する配列
  al45Vec <- c()
  al135Vec <- c()

  # 何列目か
  col_num <- 1

  # ボクセルの数だけ繰り返す
  for(i in 1:(nrow(taus)/runNum)){

    for(j in 1:runNum){

      messe <- paste('----- excution Voxel', i, '-Run', j , ' ---')
      print(messe)

      # i番目のボクセルデータ
      voxel_al45 <- al45[col_num]
      voxel_al135 <- al135[col_num]

      # i番目のボクセルの時間遅れτ
      tau_al45 <- taus[col_num, 1]
      tau_al135 <- taus[col_num, 2]

      attractor_al45 <- Attractor(voxel_al45, tau_al45, i, j, "angledlinevid-45degrees")
      attractor_al135 <- Attractor(voxel_al135, tau_al135, i, j, "angledlinevid-135degrees")

      print("al45 vectorize")
      al45Vec <- rbind(al45Vec, TDAvec(attractor_al45, i, j, "angledlinevid-45degrees"))

      print("al135 vectorize")
      al135Vec <- rbind(al135Vec, TDAvec(attractor_al135, i, j, "angledlinevid-135degrees"))

      col_num <- col_num + 1

    }

  }


  # csv書き出し
  PATH_al45Vec <- paste(PATH, 'TDAvec_autocor_al45_', parameters, '.csv', sep = "")
  write.csv(as.data.frame(al45Vec), PATH_al45Vec, quote = FALSE, row.names = FALSE)

  PATH_al135Vec <- paste(PATH, 'TDAvec_autocor_al135_', parameters, '.csv', sep = "")
  write.csv(as.data.frame(al135Vec), PATH_al135Vec, quote = FALSE, row.names = FALSE)

}


# Attractor関数
# 引数としてcol_num番目のボクセルデータをVoxel，時間遅れτをtau，何番目のボクセルかをvoxel_no，何回目の試行数かをrun_no，タスク名をtaskで受けとる
# * 時間遅れτを使って，ある時刻tの値，t+τの値，t+2*τの値で3次元データを作る
# * 3次元データを返す

Attractor <- function(voxel, tau, voxel_no, run_no, task){

  # データをずらすことで長さが変わるので注意！

  # 元データ
  x <- voxel[1:(nrow(voxel) - (2*tau)), 1]
  # 元データからτ分ずらしたデータ
  y <- voxel[(1 + tau):(nrow(voxel) - (tau)), 1]
  # 元データから2*τ分ずらしたデータ
  z <- voxel[(1 + (2*tau)):nrow(voxel), 1]

  # 3次元データとして結合
  xyz <- cbind(x, y, z)

  # アトラクタ図を出力する場合
  if (atrct_output == 1){

    # 出力するアトラクタ図を保存するのディレクトリ名・パス
    DIR_attractor <- paste(PATH, atrctName, sep="")
    PATH_attractor <- paste(DIR_attractor, '/', sep="")

    # すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
    if(!file.exists(PATH_attractor)) {
      dir.create(DIR_attractor)
    }

    # この後にplotするアトラクタ図のタイトル
    graph_name <- paste("Mapping to 3dim space : ", task, "-Voxel", voxel_no, "-Run", run_no,sep="")

    # この後に出力するpngファイル名
    PATH_graph <- paste(PATH_attractor, task, "-Voxel", voxel_no, '-Run', run_no, '.png', sep="")

    # 3次元データをplot，出力
    png(PATH_graph)
    scatterplot3d(xyz, xlab = "x = t", ylab = "y = t + τ", zlab = "z = t + 2*τ", pch = 16, type="o", main = graph_name)
    dev.off()

    print(PATH_graph)

  }

  return (xyz)

}

# TDAvec関数
# 引数としてcol_num番目のボクセルデータを3次元データにしたものをattractor，何番目のボクセルかをvoxel_no，何回目の試行数かをrun_no，タスク名をtaskで受けとる
# * ripsDiagで3次元データにTDAのPersistent Homologyを適用
# * 各次の穴情報それぞれに対してBattiNumberCount関数を使って穴の数を数え，横結合することでベクトル化
# * ベクトル化したデータを返す

TDAvec <- function(attractor, voxel_no, run_no, task){

  # TDAのPersistent Homologyを適用
  tda <- ripsDiag(X = attractor, maxdimension = 2, maxscale = ms)

  if (barcode_output == 1){

    # バーコードを保存するディレクトリ名・パス
    DIR_tda <- paste(PATH, barcodeName, sep="")
    PATH_tda <- paste(DIR_tda, '/', sep="")

    # すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
    if(!file.exists(PATH_tda)) {
      dir.create(DIR_tda)
    }

    # この後でplotするバーコードのタイトル
    barcode_name <- paste("Barcode Diagram (TDA) : ", task, "-Voxel", voxel_no, '-Run', run_no, '.png', sep="")

    # この後で出力するpngファイル名
    PATH_barcode <- paste(PATH_tda, task, "-Voxel", voxel_no, '-Run', run_no, '.png', sep="")

    # バーコードをplot，出力
    png(PATH_barcode)
    plot(tda$diagram, barcode = TRUE, main = barcode_name)
    dev.off()

    print(PATH_barcode)

  }

  # 穴情報を抽出
  df_tda <- as.data.frame(tda$diagram[, 1:3])

  # 各次の穴情報を分割
  zeroDim <- subset(df_tda, df_tda$dimension == 0)
  oneDim <- subset(df_tda, df_tda$dimension == 1)
  twoDim <- subset(df_tda, df_tda$dimension == 2)


  # 各次の穴の数を数え横結合することでベクトル化
  tdaVec <- c(BettiNumberCount(zeroDim), BettiNumberCount(oneDim), BettiNumberCount(twoDim))

  return(tdaVec)

}

# BettiNumberCount関数
# 引数として各次の穴情報をholeで受け取る
# * 穴を数える回数（kizamiNum）などのパラメータを決める
# * 穴情報はそれぞれの穴発生時の直径（Birth），穴消滅時の直径（Death）が記録されており，ある時刻timeの時の穴の数を数える
# * 1×kizamiNumのデータを返す

BettiNumberCount <- function(hole){


  # 穴をkizamiNum回数えるために時間幅を求める
  # もともとの時間はms，kizamiNumで割ることでどれぐらいずつ時刻timeをずらせばいいかわかる
  kizamiWidth <- ms/kizamiNum

  # 時刻
  time <- 0

  # 穴を数えた回数
  k <- 0

  # kizamiNum回数えた結果を格納する配列
  bettiNumbers <- numeric(kizamiNum)

  # 穴が発生していればTrue
  if(nrow(hole) >= 1){

    # kizamiNumber回穴の数を数えるまでループ
    while(k != kizamiNum){

      # 時刻timeの時の穴の数
      bettiCount <- 0

      # 発生したそれぞれの穴に対して調べる
      for(j in 1:nrow(hole)){

        # 時刻timeがある穴の発生時間中（Birth <= time <= Death）であればbettiCountに1足す
        if((hole$Birth[j] <= time) && (time <= hole$Death[j])){

          bettiCount = bettiCount + 1

        }

      }

      # bettiCountを配列に格納
      bettiNumbers[k] <- bettiCount

      # 時刻timeをずらす
      time = time + kizamiWidth

      # 穴の数を数えたのでkに1足す
      k = k + 1

    }

  } else {

    # そもそも穴が発生していなければ0をkizamiNum個格納
    bettiNumbers <- numeric(kizamiNum)

  }

  return(bettiNumbers)

}

# Execute main function
main()
