#Main
from helper import executeable_control_loop

#Executing this file runs a full loop with the set hyperparameters on the chosen data set.
#This generates a folder with the results of the experiment titled UOT - data_set - k=k_reg_m=reg_m_reg=reg.
#This folder contains information about the training loop, but, of main importance, the atoms and pdfs with saved images of the 
#labeled data set which can be comapred to gt.pdf, the ground truth.

#Explanation of variables:
#k: this is the number of dictionary atoms. While not strictly necessary, you should make this an even number.
#OT_type: the code is capable of running UBCSC and BCSC, the unbalanced and balanced loops. "UOT" gives the unbalanced and "OT" gives the balanced.
#iters: the dictionary learning loop runs for a set number of iterations which is controlled by this value. 
#reg_m: this is the marginal relaxation term in the unbalanced optimal transport problem
#reg: this is the entropic relaxation term in the unbalanced optimal transport problem
#data_set: the code base currently supports four labeled HSI data sets - "salinasA", "indian_pines", "pavia", and "paviaU"
#purity test: this boolean controls if the trial runs an accuracy test or a purity test. Given that in a purity test
#we do not know the number of ground truth classes, we, by default, run the clustering loop on [n, ..., n+10] clusters
#where n is the number of ground truth classes. 



k = 10
reg_m = 1000
reg = .1
executeable_control_loop(k = k, OT_type = "UOT", iters = 500, reg_m = reg_m, reg = reg, data_set = "salinasA", purity_test = False)