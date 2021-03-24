echo "Iterate over cross validation."

for ((ohe = 100; ohe <= 500; ohe += 100))
do
    for ((dv = 100; dv <= 500; dv += 100))
    do
        export TAG_OHE_DIM=$ohe;
        export DOCVEC_DIM=$dv;

        python3 ./topcoder_ml.py
    done
done

for metadata in 'project_id' 'duration' 'sub_track'
do
    python3 ./topcoder_ml.py --exclude-metadata $metadata;
done

for global_context in 'num_of_competing_challenges' 'competing_same_proj' 'competing_same_sub_track' 'competing_avg_overlapping_tags'
do
    python3 ./topcoder_ml.py --exclude-global-context $global_context;
done