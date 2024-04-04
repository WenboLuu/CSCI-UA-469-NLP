@echo off
echo Compiling Java files...
javac -cp maxent-3.0.0.jar;trove.jar *.java

echo Training model...
java -cp .;maxent-3.0.0.jar;trove.jar MEtrain train.features model.chunk

echo Tagging with model...
java -cp .;maxent-3.0.0.jar;trove.jar MEtag dev.features model.chunk response.chunk

echo Scoring the model...
python score.chunk.py WSJ_24.pos-chunk response.chunk

pause