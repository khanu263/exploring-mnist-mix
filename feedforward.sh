#!/bin/bash
for testing in $(eval echo "data/splits/[0-9]*test.split"); do	
	test_name=${testing:14}
	test_name=${test_name//_test.split}
	training=${testing}
	training=${training//_test.split}
	training=${training}_train.split
	python3 main.py --create feedforward 250 250 --labels agnostic --train $training 150 32 0.1 0.9 --log log/mixedff2_${test_name}.txt --save model/mixedff2${test_name}.pt --"test" $testing "test/ff2_${test_name}.txt" --gpu
done
