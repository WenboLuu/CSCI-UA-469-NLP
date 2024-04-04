@echo off
echo Compiling Java files...
javac -cp maxent-3.0.0.jar;trove.jar *.java

echo Training model...
java -cp .;maxent-3.0.0.jar;trove.jar MEtrain ../data/features/train.features model.chunk

echo Tagging with model...
java -cp .;maxent-3.0.0.jar;trove.jar MEtag ../data/features/dev.features model.chunk ../data/output/response.chunk

echo Scoring the model...
python score.chunk.py ../data/WSJ_24.pos-chunk ../data/output/response.chunk

pause