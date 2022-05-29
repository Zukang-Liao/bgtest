# bgtest

1. Download the [Background Challenge Database](https://www.dropbox.com/s/0vv2qsc4ywb4z5v/original.tar.gz?dl=0).
2. Get the 'BGDB/ORIGINAL_DIR' correct in 'config.yaml'
3. Get the 'ConfigPath' and 'SAVE_DIR' correct in 'train.py'
4. Run the lines in 'run_this.txt'

After experiments:
1. Upload the automatically generated 'model_label.txt' file in 'SAVE_DIR'
2. Do NOT delete the trained model or the logs in 'SAVE_DIR', And do NOT upload the model




Next step:
1. Download the [graph.npy](https://drive.google.com/drive/u/0/folders/1pM8Er3xVfHL1fl5e2KOMwd8CLtqdhaeQ).
2. Download the databases from [here](https://drive.google.com/drive/u/0/folders/1Qyh0_kOOq-lvXSOHt5i8q-yTwP5A6csI).
3. Download the pre-saved itemsets from [here](https://drive.google.com/drive/u/0/folders/1GHmzBLKQTDDYESn24K5oVoqZePJOgRPa)
<!-- 3. Download the pretrained segmentation tools from [here](https://drive.google.com/drive/u/0/folders/1pM8Er3xVfHL1fl5e2KOMwd8CLtqdhaeQ). -->
4. Change the path to the 'CAND/graph_path' in 'config.yaml'
5. Change the paths to the databases in 'config.yaml', namely 'BGDB/BG_T_DIR', 'BGDB/ORIGINAL_DIR', 'BGDB/MASK_DIR', 'BGDB/NO_FG_DIR' and 'BGDB/BG20K_DIR'.
<!-- 6. Change the paths to the segmentation tools in 'config.yaml', namely 'SCENE/scene_model', 'SEGMENT/encoder_path', and 'SEGMENT/decoder_path'. -->
6. Change the paths to the items in 'config.yaml', namely 'CAND/bgdb_items' and 'CAND/bg20k_items'


New:
1. Download the [candidx.npy](https://drive.google.com/drive/u/0/folders/1pM8Er3xVfHL1fl5e2KOMwd8CLtqdhaeQ)
2. Change the path to the 'CAND/CANDIDX_PATH' in 'config.yaml'
