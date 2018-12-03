PATH_DATA="../../Data_mri/angledlinevid-1fe/"
PATH_SAVE="../"

for dir in 20181119mt 20181119tsk 20181119tst
do


  PATH_NII="${PATH_DATA}${dir}/"


  PATH_BA="${PATH_SAVE}MaskBrodmann/${dir}/"

  echo "------------ ${PATH_NII} | ${PATH_BA} ---------------"

  python Preprocessing_nii2zscore.py ${PATH_NII} ${PATH_BA} rwmaskBA.nii




done
