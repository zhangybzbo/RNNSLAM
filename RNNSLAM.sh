name=~/projects/ColonData/sequences/
num=031
./build/bin/dso_dataset \
	mode=2 preset=0 \
	files=$name$num'/image' \
	calib=$name'calib_270_216.txt' \
	rnn=~/projects/RNNSLAM/src/RNN \
	rnnmodel=~/projects/ColonData/RNNmodel/model-145000 \
	numRNNBootstrap=9 lostTolerance=5 \
	output_prefix=$name$num'/reproduce/' \
	quiet=1 sampleoutput=1

