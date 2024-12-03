@echo off
set TRANSFORMERS_OFFLINE=1
set "data_path=..\datasets"
set "ann_path=..\clip_train"
set "train_image_root=cc3m"
set "epochs=30"
set "data=cc3m"
set "gamma=0.8"
set "rho=6.0"
set "ita_type=sogclr"
set "opt=adamW"
set "train_file=%data%_train_subset.json"
set "saved_model=..\output\final\%ita_type%_%opt%"

set "CUDA_VISIBLE_DEVICES=0"
set "zs_dataset=imagenet"

python clip.py ^
    --data_path "%data_path%" ^
    --ann_path "%ann_path%" ^
    --train_file "%train_file%" ^
    --ita_type %ita_type% ^
    --train_image_root "cc3m_subset_100k/" ^
    --output_dir "%saved_model%" ^
    --evaluate ^
    --checkpoint "%saved_model%\checkpoint_1.pth" ^
    --zs_datafolder "..\datasets\imagenet\val" ^
    --use_amp ^
    --zs_dataset "%zs_dataset%" ^
    --device cuda:0 ^
    --no-distributed > "%data%_%ita_type%_%opt%_g%gamma%_rho%rho%_e%epochs%_val_ck1.log"
pause