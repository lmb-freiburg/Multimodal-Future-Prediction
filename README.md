This repository corresponds to the official source code of the CVPR 2019 paper:

<a href="https://arxiv.org/pdf/1906.03631.pdf">Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction</a>

To get an overview about the method and its results, we highly recommend checking our poster and a short video at <a href="https://lmb.informatik.uni-freiburg.de/Publications/2019/MICB19/">[Page]</a>


![demo](demo.gif)


#### Requirements

- Tensorflow-gpu 1.14.
- opencv-python, sklearn, matplotlib (via pip).

#### Setup
We use the source code from WEMD[1] to compute our SEMD evaluation metric.

- extract the blitz++.zip under /wemd.
- cd build
- cmake ..
- make

After compilation, you should get a library under /wemd/lib, which is linked in the wemd.py.

#### Data

To reproduce our results in the paper, we provide the processed testing samples from SDD [2] used in our paper. Please download them from <a href="https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/datasets.zip">[Link]</a>

After extracting the datasets.zip, you will get a set of folders representing the testing scenes. For each scene you have the following structure:

- imgs: contains the images of the scene.
- floats: for each image, we store -features.float3 and -labels.float3 files. The former is a numpy array of shape (1154,5) which can store up to 1154 annotated objects. Each object has 5 components describing its bounding box (tl_x, tl_y, br_x, br_y, class_id). The indexes of the objects represent the tracking id and are given in the file -labels.float3.
- scene.txt: each line represent one testing sequence and has the following format: tracking_id img_0,img_1,img_2,img_future.

#### Models

We provide the final trained model for our EWTAD-MDF. Please download them from <a href="https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/models.zip">[Link]</a>

#### Testing

To test our EWTAD-MDF, you can run:

python test.py --output

- --output: will write the output files to the disk under the path specified in the config.py (OUTPUT_FOLDER_FLN). If you need only to get the testing accuracies without writing files (much faster), you can simply remove the --output.

#### Training

We provide additionally the loss functions used when training our sampling-fitting network, please check the net.py file for more details.


#### Citation

If you use our repository or find it useful in your research, please cite the following paper:


<pre class='bibtex'>
@InProceedings{MICB19,
  author       = "O. Makansi and E. Ilg and {\"O}. {\c{C}}i{\c{c}}ek and T. Brox",
  title        = "Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction",
  booktitle    = "IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = " ",
  year         = "2019",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2019/MICB19"
}
</pre>

#### References

[1] S. Shirdhonkar and D. W. Jacobs. Approximate earth movers distance in linear time. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8, June 2008.

[2] A. Robicquet, A. Sadeghian, A. Alahi, S. Savarese, Learning Social Etiquette: Human Trajectory Prediction In Crowded Scenes in European Conference on Computer Vision (ECCV), 2016.

#### License

![logo](logo-header.png)

This source code is shared under the license CC-BY-NC-SA, please refer to the LICENSE file for more information.

This source code is only shared for R&D or evaluation of this model on user database.

Any commercial utilization is strictly forbidden.

For any utilization with a commercial goal, please contact [contact_cs](mailto:contact_cs@imra-europe.com) or [bendahan](mailto:bendahan@imra-europe.com)
