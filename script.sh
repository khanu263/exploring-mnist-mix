#!/bin/bash
for testing in $(eval echo "data/splits/[0-9]*test.split"); do
	for model in $(eval echo "model/resnet4?*.pt"); do
		model_name=${model:13}
		model_name=${model_name//.pt}
		test_name=${testing:14}
		test_name=${test_name//_test.split}
		python3 main.py "--load" ${model} --labels agnostic --"test" $testing "test/${model_name}_tested_on_${test_name}_resnet4_model.txt" --gpu
	done
done	

