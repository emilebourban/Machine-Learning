# Machine-Learning: Project 1



run.py

	We use this file to make our final prediction with the best accuracy. It uses ridge regression at degree 10. 
	It also splits the data according to the 4 categories given by column 22, then applies the ridge reg to the separated data
	and fits 4 weight for every category. The prediction data is also split in the same way and the weights are applied to 
	it according to its category. It is then merges back together and submited in "sample_submission.csv"
	
main.py

	Here, you can find exemples of the applications of every fitting method that we train our model with. In every case a cross validation was daone
	with a 20/80 split. The boolean parameters at the begining allows the user to only execute certain parts of the program since it takes a long time.
	In cases where it applies, the DEGREE list can be used to run the code on any chosen degree.
	
	GADIENT DESCENT: (GRADIENT_DESCENT set to True)
		Predicts the weights on the data using the iterative gradient descent method, takes a long time
	
	STOCHASTIC GRADIENT DESCENT: (STOCHASTIC_GRADIENT_DESCENT set to True)
		Same principle as above exept the gradient is computed on minibatchs of the data to decrease complexity.
	
	LINEAR REGRESSION: (REG_LIN set to True)
		Simple linear regression on the unmodified data predicts the proportionality between the features and the outputs. The fitting
		and minimizing of the cost function is now done with the least squares method.
	
	LINEAR REGRESSION W/ OFFSET: (OFFSET set to True)
		Same as before with an offset parameter to increase accuracy.
	
	POLYNOMIAL REGRESSION: (POLY set to True)
		Now we create a data_poly matrix that has added polynomial coefficients of the original data to fit with a polynomial weights
		since the prediction probably depends on the more than a degree 1 proportionality.
	
	POLYNOMIAL RIDGE REGRESSION: (POLY_RIDGE set to True)
		Uses the polynomial data for the prediction, now, the ridge reg regularizes the data in function of the L2 norm of the weights		
	
	POLYNOMIAL RIDGE REGRESSION W/ CATEGORY SEPARATION: (RIDGE_CAT set to True)	
		Data is split according to its type, then cleaned of any useless column. 4 different weights are then trained on the model so that our 
		prediction is improved
	
	LOGARITHMIC REGRESSION: (LOG_REGRESS set to True)
		Uses logarithmic regression to predict the data, only works on degree 1
	
	REGULARISED LOGARITHMIC REGRESSION: (REG_LOG_REGRESS set to True)
		Uses regularised logarithmic regression to predict the data, only works on degree 1

		
For each hyperparameter, degree, lambdas, gamma, ... we used larger denser lin/logspaces in order to have a more precise fit. For
reasons of speed, we use a lower amount of point for the submission but the user can change this as he pleases.

Note: Our 3rd teamate (Timothée Kastylevsky) who did not participate actively to the project was supposed to make up for it and do a significant part this week-end. However, he did not do any work, notified us only sunday evening, and then told us on monday that he would probably drop out of EPFL. As a result, we did not have time to fully implement the logistic regressions. Thank you for taking note of this when grading our work (as discussed via e-mail with Pr. Jaggi)
