# Nexperia_Project_Jasper
This is a repo including codes related to Nexperia project, which is created by Jasper

## Folder Description:

### "focal_loss_related" 

 this folder only contains the codes to train Batch2

### "NexperiaTrainTxt" 

this folder contains the codes related to Nexperia_compare_trainingSet.

The structure of the folder is as follows:

```
SavedModel (contains the trained models)

Src
	test_model
		(contains the codes to test model on different datasets)
	train_model
		(contains the codes to train model)
	train_model_imbalance_sampler
		(contains the codes to implement imbalanced sampler)

useful_function
	calculate_mean_std.py
		(the code to calculate mean&std of input images)
	extract_noisy_data.py
		(the code to plot bad_score distribution and extract the possible noisy images in test data)
	img_similarity_removeEmpty.py
		(the code to remove the possible noisy data according to hash value)
	
```

