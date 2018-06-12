# Group-Equivariant Convolutional Neural networks for Keras: keras_gcnn
Straight-forward keras implementations for 90-degree roto-reflections equivariant CNNs. See a [working example](https://github.com/basveeling/keras-gcnn/blob/master/examples/g_densnet_cifar.py).

Install: `pip install git+https://github.com/nom/GrouPy#egg=GrouPy -e git+https://github.com/basveeling/keras-gcnn.git#egg=keras_gcnn`

Requires python 3, up to date keras and a tensorflow backend.
Please report any problems in the issues.

## About Group-equivariance
Conventional fully-convolutional NNs are 'equivariant' to translation: as the input shifts in the spatial plane, the output shifts accordingly. This can be extended to include other forms of transformations such as 90 degree rotations and reflection. This is formalized by [2].

## Citing
If you use these implementations in your work, we appreciate a citation to our paper:

**[1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". [arXiv:1806.03962](http://arxiv.org/abs/1806.03962)**

Biblatex entry:
```latex
@ARTICLE{Veeling2018-qh,
  title         = "Rotation Equivariant {CNNs} for Digital Pathology",
  author        = "Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and
                   Cohen, Taco and Welling, Max",
  month         =  jun,
  year          =  2018,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "1806.03962"
}
```

## GDensenet
![GDensenet](https://github.com/basveeling/keras-gcnn/blob/master/model.png)
We provide a Group-equivariant version of DenseNet [3] as proposed in [1].

## Recipe for building equivariant networks:
- Decide on a group to use, currently D4 (roto-reflection) and C4 (rotations) are supported.
- All convolutional layers with kernels larger than 1 should be replaced with group-equivariant layers.
    - The first layer transforms the input from Z2 to D4, by setting `h_input='Z2'` and `h_output='C4'` or `'D4'`.
    - Follow up layers live on the chosen group and have `h_input=h_output='D4'` (or `'C4'`).
- Operations that learn parameters per feature-map should be replaced with group versions, including:
    - BatchNormalization becomes GBatchNorm.
- To create a model invariant to rotations, use GroupPool followed by a global spatial pooling layer such as GlobalAveragePooling.

## References
- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv [cs.CV] (2018), (available at http://arxiv.org/abs/1806.03962).
- [2] Cohen, Taco, and Max Welling. "Group equivariant convolutional networks." International Conference on Machine Learning. 2016.
- [3] Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. Vol. 1. No. 2. 2017.
