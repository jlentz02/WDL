# Unbalanced Optimal Transport Barycenters for Unsupervised Clustering of Hyperspectral Data
This project leverages a modification of Wasserstein dictionary learning to create a low dimensional representation of HSI data. We perform spectral clustering on the resulting weights matrix to generate labels of the HSI data. 

---

## Description 
This project improved upon prior work by Fullenbaum, et al (2024) by incorporating unbalanced optimal transport barycenters in the Wasserstein dictionary learning loop. By doing so, we lift the restriction of having balanced distributions, and no longer need to scale each HSI pixel to be total mass one. This enables a more faithful representation of the underlying data as the total mass of each HSI pixel is now incorporated into the dictionary learning process. This modification has resulted in improved labeling accuracy when compared to the balanced approach. This code base maintains functionality for running tests using balanced optimal transport. 
This project is intended for other researchers, and people who are interested in applications of dictionary learning. 

---

## Installation
# Clone the repo
git clone https://github.com/jlentz02/WDL
cd WDL

# Install dependencies
pip install -r requirements.txt

---

## Usage
To run the code, it is necessary to navigate your directory to \WDL\WDL_HSI\WDL_HSI-main\Tests. You can then execute main.py with arguments of your choice. These are explained in more detail in the main, but the primary controls are k, the number of atoms, OT_type, the choice between OT or UOT barycenters, reg_m, the marginal relaxation term, reg, the entropic regularization term, data set, and purity. The latter controls whether the test is an accuracy test, or a purity test. 

In addition to the dependencies described above, it is necessary to download the [data_set].mat and [data_set_gt].mat files from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes. For the Salinas A and Indian pines data sets, download the _correct.mat versions of the data. These have water reflectance, and other broken bands removed. Unfortunately, these files are too large to be uploaded to github. There is no need to rename the files. The code is structured in such a way to read the differing file names. However, this will not be the case if additional data sets are added. 

At present, only the four data sets including in our paper, Salinas A, Pavia University, Pavia Centre, and Indian Pines are supported by the code. If you wish to run this code yourself on additional data sets, it will be necessary to have data and ground truth .mat files formatted correctly. You will then need to append the wdl_instance function in helper.py with the structure of the fname and matname. It will also be necessary to hardcode a dictionary in the clustering_loop function assigning ground truth classes for the coloring during image generation. 

If you have any specific questions, please feel free to contact me at joshua.lentz12@gmail.com, and I will do my best to help. 

## License
This project is protected under the MIT License https://mit-license.org. 

