MINICONDA_DIR=/software/ac18804/miniconda3
ML_SOFTWARE=
source $MINICONDA_DIR/etc/profile.d/conda.sh
conda activate ml_env
cd $ML_SOFTWARE
export PYTHONNOUSERSITE=true
export PYTHONPATH="$PWD:$PYTHONPATH"
