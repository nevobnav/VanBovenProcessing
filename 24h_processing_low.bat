:START
@echo on
cd /d "C:\Users\VanBoven\Anaconda3\Scripts"
call activate.bat VanBoven3.6
python "C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing\prepare_processing_files.py"
cd "C:\Program Files\Agisoft\Metashape Pro"
call metashape.exe -r "C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing\Batch_processing.py" "Low"
call metashape.exe -r "C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing\archive_psx_files.py"
TIMEOUT /T 1800
::GOTO START
