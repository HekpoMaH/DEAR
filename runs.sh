for algo in  "insertion_sort" 
do
    for seed in 1 2 
	do

	    #python train_reasoner.py --algorithms $algo --config-path baseline.yaml
	    python train_reasoner.py --algorithms $algo --config-path deq.yaml &
	    #python train_reasoner.py --algorithms $algo --config-path deq_alignment.yaml
	done 
done
