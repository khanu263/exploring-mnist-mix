#!/bin/bash
python3 main.py --create resnet 4 4 4 --labels agnostic --train "data/splits/0_arabic_train.split" 1 32 0.01 0.9 --log log/mixedres4_arabic_epoch0.txt --save model/mixedres4.pt --test "data/splits/0_arabic_test.split" "test/resnet4mix_arabic_epoch0.txt" --gpu
for testing in $(eval echo "data/splits/[1-9]*test.split"); do
	test_name=${testing:14}
	test_name=${test_name//_test.split}
	training=${testing}
	training=${training//_test.split}
	training=${training}_train.split
	
	python3 main.py "--load" model/mixedres4.pt --labels agnostic --train $training 1 32 0.01 0.9 --log log/mixedres4_${test_name}_epoch0.txt --save model/mixedres4.pt --"test" $testing "test/resnet4mix_${test_name}_epoch0.txt" --gpu
done	

python3 main.py --load model/mixedres4.pt --labels agnostic --test data/splits/all_test.split "test/resnet4mix_all_epoch0.txt" --gpu
for i in $(eval echo {1..200}); do

	for testing in $(eval echo "data/splits/[0-9]*test.split"); do
		test_name=${testing:14}
		test_name=${test_name//_test.split}
		training=${testing}
		training=${training//_test.split}
		training=${training}_train.split
	
		python3 main.py "--load" model/mixedres4.pt --labels agnostic --train $training 1 32 0.01 0.9 --log log/mixedres4_${test_name}_epoch$i.txt --save model/mixedres4.pt --"test" $testing "test/resnet4mix_${test_name}_epoch$i.txt" --gpu
	done	
	
	python3 main.py --load model/mixedres4.pt --labels agnostic --test data/splits/all_test.split "test/resnet4mix_all_epoch$i.txt" --gpu
done
python3 main.py --load model/mixedres4.pt --labels agnostic --test data/splits/all_test.split "test/resnet4mix_all.txt" --gpu

