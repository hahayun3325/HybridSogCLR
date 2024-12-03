@echo off
rem Setting environment variables
set PYTHONPATH=%PYTHONPATH%;.\bimodal_exps
set HUGGINGFACE_HUB_CACHE=.\checkpoints\huggingface
rem Explicitly enable online mode
set TRANSFORMERS_OFFLINE=0
set HF_HUB_OFFLINE=0
set HF_DATASETS_OFFLINE=0

rem Create cache directory if it doesn't exist
if not exist ".\checkpoints\huggingface" mkdir ".\checkpoints\huggingface"

rem Download the model first
echo Downloading required models...
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./checkpoints/huggingface'); AutoModel.from_pretrained('distilbert-base-uncased', cache_dir='./checkpoints/huggingface')"
if errorlevel 1 (
    echo Failed to download models. Please check your internet connection.
    pause
    exit /b
)

rem Setting training parameters
set data_path=..\datasets
set ann_path=..\clip_train
set train_image_root=cc3m_subset_100k/
set data=cc3m
set train_file=%ann_path%\%data%_train_subset.json
set gamma=0.8
set rho=6.0
set epochs=30
set ita_type=hybrid
set opt=adamW
set CUDA_VISIBLE_DEVICES=0

python clip.py ^
    --data_path %data_path% ^
    --ann_path %ann_path% ^
    --train_file %train_file% ^
    --train_image_root %train_image_root% ^
    --output_dir ..\output\%ita_type%_%opt%_%data%_g%gamma%_rho%rho%_e%epochs% ^
    --init_model ^
    --use_amp ^
    --ita_type %ita_type% ^
    --opt %opt% ^
    --tau_init 0.01 ^
    --sogclr_gamma %gamma% ^
    --rho_I %rho% ^
    --rho_T %rho% ^
    --eta_init 0.03 ^
    --sched cosine ^
    --epochs %epochs% ^
    --text_encoder distilbert-base-uncased ^
    --no-distributed > %data%_%ita_type%_%opt%_g%gamma%_rho%rho%_e%epochs%.log

pause