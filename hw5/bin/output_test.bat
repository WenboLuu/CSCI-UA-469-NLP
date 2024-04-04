@echo off
echo Tagging with model for test set...
java -cp .;maxent-3.0.0.jar;trove.jar MEtag test.features model.chunk WSJ_23.chunk.chunk

pause