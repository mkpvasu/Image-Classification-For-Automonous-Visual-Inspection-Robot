# Image-Classification-For-Automonous-Visual-Inspection-Robot

* Implemented a ResNet50 architecture for Multi-class Image Classification task for Automonous Visual Inspection Robot to be used for production line 
inspection of aerospace components at GKN Aerospace facilities.
* Prepared custom dataset containing approximately around 500 images (class balanced for 3 labels) of 9504 * 6336 resolution for each micron sizes of 5, 10 and 20.
* The images were categorized into 3 different class labels of Good, Marginal and Bad depending on defects present in captured image at a particular area of the 
part after sanding.
* Trained individual models for 3 microns sizes with modified hyperparameters achieving a F1 score of o.55, o.64 and 0.73 for 5, 10 and 20 microns after 30 epochs.
* Devised strategies to improvise the model performances including generating more good data, hyperparmeter tuning and using different optimizers and learning rate 
schedulers to achieve higher performances.
