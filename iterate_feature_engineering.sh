echo "Feature engineering with default dimension...\c"
python3 ./topcoder_mongo.py
echo "done";

echo "Building challenge tag combination with different OHE dimension..."
for (( ohe = 100; ohe <= 500; ohe += 100 ))
do
    export TAG_OHE_DIM=$ohe;
    echo "Training TAG_OHE_DIM=$TAG_OHE_DIM...\c";
    python3 -c "import topcoder_mongo as DB; DB.TopcoderMongo.write_tag_feature();";
    echo "Done";
done
echo "Challenge tag done";

