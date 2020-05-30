while true  
do  
	echo
	echo "######################################################################"
	echo $(date)
  for i in {0..9}
	do
	kaggle kernels status gabzchua/fbprophet-$i
	done 
  sleep 300  
done

