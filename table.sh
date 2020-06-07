#!/bin/bash
languages={arabic,bangla,devanagari,english,farsi,kannada,swedish,telegu,tibetan,urdu}
rm table_res.txt
echo "Rows are tested languages. Columns are trained models of language" >> table_res.txt
for model in $(eval echo $languages); do
	for testing in $(eval echo $languages); do
		tail -1 "test/${testing}_tested_on_${model}"* >> table_res.txt
		printf %s " " >> table_res.txt
	done
	echo " " >> table_res.txt
done	
