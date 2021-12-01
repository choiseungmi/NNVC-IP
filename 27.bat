set root=C:\Users\user\anaconda3
call %root%\Scripts\activate.bat %root% 

call conda env list 
call conda activate cap2
call cd C:\Users\user\Desktop\VVCSoftware_VTM-VTM-9.0
call python train.py --epochs 1000 -lr 1e-2 --batch-size 16 -q 27 -hgt 32 -wdt 32 --cuda --save

pause