@echo off
echo Tagging with model for test set...
java -cp .;maxent-3.0.0.jar;trove.jar MEtag ../data/features/test.features model.chunk ../data/output/WSJ_23.chunk

pause