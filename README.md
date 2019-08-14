# Hyper-parameter optimization in deep learning and transfer learning - Applications to medical imaging

This repo contains my PhD thesis. It was done between 2015 and 2018, and was defended the 15/01/2019 at Télécom ParisTech.

The PhD was done as a collaboration between Télécom ParisTech and Philips Healthcare, my supervisors were Isabelle Bloch, Roberto Ardon and Matthieu Perrot.

Here is the abstract:
In the last few years, deep learning has changed irrevocably the field of computer vision. Faster, giving better results, and requiring a lower degree of expertise to use than traditional computer vision methods, deep learning has become ubiquitous in every imaging application. This includes medical imaging applications. 

At the beginning of this thesis, there was still a strong lack of tools and understanding of how to build efficient neural networks for specific tasks. Thus this thesis first focused on the topic of hyper-parameter optimization for deep neural networks, i.e. methods for automatically finding efficient neural networks on specific tasks. The thesis includes a comparison of different methods, a performance improvement of one of these methods, Bayesian optimization, and the proposal of a new method of hyper-parameter optimization by combining two existing methods: Bayesian optimization and Hyperband.

From there, we used these methods for medical imaging applications such as the classification of field-of-view in MRI, and the segmentation of the kidney in 3D ultrasound images across two populations of patients. This last task required the development of a new transfer learning method based on the modification of the source network by adding new geometric and intensity transformation layers.

Finally this thesis loops back to older computer vision methods, and we propose a new segmentation algorithm combining template deformation and deep learning. We show how to use a neural network to predict global and local transformations without requiring the ground-truth of these transformations. The method is validated on the task of kidney segmentation in 3D US images. 
