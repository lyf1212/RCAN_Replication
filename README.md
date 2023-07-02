# RCAN_Replication
- This repository loads four basic python file to repliate the classic SISR end2end model, RCAN.
- More specofically,
  - I use 'model.py' to build RCAN model, which contains the important RIR struct and Channel Attention block;
  - 'train.py' to build a basic pytorch deep learning training pipeline; 'dataset.py' to manipulate the DIV2K dataset;
  - 'preprocess_subimages.py' as well as 'extract_subimages.py' to preprocess those rather big to regard as input images;
  - finally, 'utils.py' contains some useful functions.
- I use 2 days to code and 1 day to train the model on one RTX2080Ti, with a just converge result which you can see in 'loss.png'.
