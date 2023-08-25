Readme

Overview:
This project consists of the following files:
	0.0create_masks.ipynb
	0.1create_masks_turkey.ipynb
	1.model_train.ipynb
	2.model_finetuning.ipynb
	3.model_eval.ipynb

Preprocessing:
The notebooks 0.0create_masks.ipynb and 0.1create_masks_turkey.ipynb are used to generate masks from polygon annotations. These masks are required for Model Training for building detection using satellite images. 
	To use the mask generation code using 0.0create_masks.ipynb, set the input and output paths.
	To use the mask generation code using 0.1create_masks_turkey.ipynb, set the paths to the GeoJSON (jsonpath), TIFF directory (tiffpath) and the output_path.

Model Training:
The notebook 1.model_train.ipynb is used to train a UNet model to predict which pixels in satellite images belong to building footprints and which don't. Mexico satellite images and masks have been used for model training. The input data is split into train, test and validation sets. The BuildingTrainer is used to define the training step, validation_step and the optimizer. 
	To run this file, set the 'DATA_PATH' variable to the folder containing the training data. 
	
Model Finetuning:
The notebook 2.model_finetuning.ipynb is used to finetune the previously trained model for continual learning. A subset of Turkey satellite images and masks have been used for this purpose. The BuildingTrainer is used to define the training step, validation_step and the optimizer. 
	To run this file, set the 'DATA_PATH' variable to the folder containing the data for finetuning.
	
Model Evaluation:
The notebook 3.model_eval.ipynb is used to evaluate the performance of the trained and tuned models on Turkey dataset. The model performance is reported in terms of average IoU score.
 	To run this file, set the 'DATA_PATH' variable to the folder containing the evaluation data.
