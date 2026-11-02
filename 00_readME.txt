This is an adaption of the published code by Vong and Lake (2022). 
To execute successfully, rename one of the raw_images_xxx folders in data to raw_images.


Key changes are:

- scripts need to be run manually, implemented to reduce inter-script dependencies for adaptions, original called all scripts in different parts of main

- all experiments except for the zipfian and referential ambiguity were removed

- added image pre-processing script to accommodate different dataset

- resizing is handles in image pre-processing instead of feature extraction 

- experiments created by the data script are now saved so they can be reused when comparing models


NOTE: Some of the commenting and addition of checkpoint print statements were added using AI.


Description of the scripts:


0_preprocessing_images
	- added to return images with a similar composition to the NOUN dataset

	- finds the shape of the object with morphological closing (if not found: take original image, resize, add white border)
	- crop image to bounding box around the detected image
	- resize resulting image so longest borders of the bounding box become 204px long (uses cv2.INTER_AREA interpolation)
	- center the resulting image on 224 x 224


1_feature_extraction
	- Creates an array with the feature vectors and a txt with the image labels:

	- loads pretrained VGG-16 model, removes final classification layers, freezes weights
    	- load images and normalize with imagenet statistics
    	- passes images through model and saves them as pickle file (4096 dim vector)
    	- extract labels from image names (naming convention: <category>_<description>__<source_dataset>)
    	- saving labels under image_labels.txt (for compatibility with vong and lake scene generation)


2_data
	- generates experiments with training and evaluation datasets

   	- training & test: combinations of objects and words as specified in the experiment configuration
	- saves the experiments as .pt file so the same set of scenes can be used for several models


3_main
	- conducts experiments with the created training and evaluation datasets

    	- saves csv under results
    	- Loss functions all moved into one class, SGCL was added
    	- only referential ambiguity and zipfian experiments were kept
    	- sampling function was altered to reduce code repetitions
	
	
