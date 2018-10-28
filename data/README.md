# Explanations about the provided dataset

The untarred folder includes two files (`static_particles.pkl` and `group_result_km6_aaad_sd0.pkl`) and one folder (`new_tfdata`). 

The `static_particles.pkl` contains information about 5000 static particles representing the floor. 

The `group_result_km6_aaad_sd0.pkl` is a hierarchical grouping result for the provided dataset. To generate this file for another dataset, you can run go to `scripts` folder and then run this command:

```
python generate_grouping.py --datapath path_to_your_dataset/new_tfdata --savepath path_to_your_save --cluster_alg KMeans  --cluster_kwargs {'"'n_clusters'"':6} --num_per_level 6 --rand_seed 0 --add_axis 1 --add_dist 1
```
