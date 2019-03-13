:START
cd /d "C:\Users\VanBoven\Anaconda3\Scripts"
call activate.bat VanBoven3.6
python "E:\VanBovenDrive\VanBoven MT\700 Data and Analysis\710 Data Processing\711 Scripts\Operational\test.py"
cd "C:\Program Files\Agisoft\Metashape Pro"
::call metashape.exe -r "E:\VanBovenDrive\VanBoven MT\700 Data and Analysis\710 Data Processing\711 Scripts\Operational\Batch_processing_1_0.py"
TIMEOUT /T 1800
GOTO START
