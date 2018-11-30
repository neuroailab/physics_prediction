# Explanations about the provided dataset

The untarred folder includes two files (`static_particles.pkl` and `group_result_km6_aaaddd_sd0.pkl`) and three folders (`new_tfdata`, `new_tfvaldata`, `pretrained_model`). 

The `static_particles.pkl` contains information about 5000 static particles representing the floor. 

The `group_result_km6_aaaddd_sd0.pkl` is a hierarchical grouping result for the provided dataset. To generate this file for another dataset, you can go to `scripts` folder and then run this command:

```
python generate_grouping.py --datapath path_to_your_dataset/new_tfdata --savepath path_to_your_save --cluster_alg KMeans  --cluster_kwargs {'"'n_clusters'"':6} --num_per_level 6 --rand_seed 0 --add_axis 1 --add_dist 1 --dyn_div 1
```

The `new_tfdata` contains training data and `new_tfvaldata` contains testing data. And `pretrained_model` includes a model trained using the provided training bash script on this dataset.
