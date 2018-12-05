# PATH_DATA="../../Data_mri/angledlinevid-1fe/"
# PATH_SAVE="../"
#
# for dir in 20181119tm 20181119tsk 20181119tst
# do
#
#   PATH_NII="${PATH_DATA}${dir}/"
#
#
#   PATH_BA="${PATH_SAVE}MaskBrodmann/${dir}/"
#
#   echo "------------ ${PATH_NII} | ${PATH_BA} ---------------"
#
#   python Preprocessing_nii2zscore.py ${PATH_NII} ${PATH_BA} rwmaskBA.nii
#
#
# done


PATH_DATA="../MaskBrodmann/"

# 被験者フォルダ名取得
SUBs=`ls -F ${PATH_DATA} | grep /`

for sub in $SUBs
do

  # # voxelフォルダがあるディレクトリまでのパス
  # PATH_voxel="${PATH_DATA}${sub}"
  #
  # echo "---------- ${PATH_voxel} ------------"

  # python Preprocessing_tasks.py ${PATH_voxel}

  # RawDataフォルダまでのパス
  PATH_RAW="${PATH_DATA}${sub}RawData/"

  echo "---------- ${PATH_RAW} -------------"

  # python Vec_TAUautocor.py ${PATH_RAW}
  #
  # Rscript Vec_TDAvec_autocor_custom100.r ${PATH_RAW}
  #
  # Rscript Vec_TDAvec_autocor_custom300.r ${PATH_RAW}
  #
  # python Vec_TDAvec_revec.py ${PATH_RAW}
  #
  # python ML_SVM_timeseries.py ${PATH_RAW}
  #
  # python ML_SVM_TDAautocor.py ${PATH_RAW}
  #
  # python ML_1dCNN_timeseries.py ${PATH_RAW}
  #
  # python ML_1dCNN_TDAautocor.py ${PATH_RAW}

  python ML_SVM_VOXtimeseries.py ${PATH_RAW}

done
