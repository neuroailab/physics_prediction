expId=physics_pred_better
group_file=group_result_km6_aaaddd_sd0.pkl
python train.py \
    --expId ${expId} \
    --alpha 0.01 \
    --number_of_kNN 10 --OB1 48 --OB2 49 \
    --room_center 30,0.2,30 --group_file ${group_file} \
    --with_coll 1 --max_collision_distance 0.3 \
    --gravity_term 9.81 --with_act 1 --seq_len 2 \
    --network_func physics_network \
    --batchsize 256 \
    --with_static 1 --add_gloss 4 \
    "$@"
