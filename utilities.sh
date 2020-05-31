for i in {1..9}
do 
	sed "s/part = 0/part = $i/g" fbprophet/prophet-0/fbprophet-0.py > fbprophet/prophet-$i/fbprophet-$i.py
done

for i in {1..9}
do 
	sed "s/-0/part = $i/g" fbprophet/prophet-0/kernel-metadata.json > fbprophet/prophet-$i/fbprophet-$i.py
done


for i in {0..9}
do 
	sed "s/"



for i in ca1 ca2 ca3 ca4 tx1 tx2 tx3 wi1 wi2 wi3
do 
	echo gabzchua/m5-train-$i
	mkdir lightgbm-$i
	cd lightgbm-$i
	kaggle kernels pull gabzchua/m5-train-$i -m
	kaggle kernels push
	cd ...
done

for i in {0..9}
do 
	cd stage1/prophet-$i
	kaggle kernels push
	cd ...
done



for i in ca1 ca2 ca3 ca4 tx1 tx2 tx3 wi1 wi2 wi3
do
	kaggle kernels status gabzchua/m5-train-$i
done