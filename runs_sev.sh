for algo in  "dag_shortest_paths" "dfs" "strongly_connected_components" "floyd_warshall"
do
	for seed in 1 2
	do
	    #python train_reasoner.py --algorithms $algo --config-path baseline.yaml
	    #python train_reasoner.py --algorithms $algo --config-path deq.yaml
	    python train_reasoner.py --algorithms $algo --config-path deq_alignment.yaml &
	done
	wait
done
