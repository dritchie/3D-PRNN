# 3D-PRNN
Torch implementation of ICCV 17 [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zou_3D-PRNN_Generating_Shape_ICCV_2017_paper.pdf): "3D-PRNN, Generating Shape Primitives with Recurrent Neural Networks"

<img src='figs/teasor.jpg' width=400>

## Instructions for PGP comparison
1. Get a dataset of shapes you want to train on, in JSON format. Each JSON should be a dictionary mapping part names to parts. Each part is a dictionary containing the following properties: center, xd, yd, zd, xdir, ydir, zdir. You'll want this split into a training set and a validation set (two directories)
1. Run `prepareTrainingData.lua`. Take a look at the command line args this expects. It lets you specify the directories containing the training and validation set JSONs, as well as the output directory where the formatted training data will be written. It also takes an optional argument which is the number of test time samples you want to generate. This is needed because at test time, the network has to be primed with a initial value (these initial values will be drawn from the combination of the training and validation data).
1. Run `driver_mine.lua` to train the model. Take a look at the command line args it expects. You should be able to leave all of them as their deafults (though you may want to change the `trainData` and `valData` args if you wrote the formatted training data anywhere other than `./mydata`). This will kick off a long-running training job. I usually just kill it after it seems like the validation loss isn't going down any more.
1. Run `testNet_mine.lua` to generate samples from the trained model. This expects to find the train model in `./models/model_full.t7`, which is where it will be saved by the training script, unless you moved it somewhere else. If you wrote/moved your data anywhere other than `./mydata`, then you'll also need to edit line 19 to point to that location.
1. The generated samples will be written to `./myresult` as JSON files in the same format as the input. You can run the script `json2obj.py` to convert these to OBJ meshes, or you can run `json2npz.py` to conver them to the Numpy storage format that the Blender rendering script expects.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN
- Torch
  
  matio: https://github.com/tbeu/matio
  
  distributions: https://github.com/deepmind/torch-distributions

- Matlab (for visualization)

## Data
- Download primitive data to current folder
```
wget http://czou4.web.engr.illinois.edu/data/data_3dp.zip
```
  
This includes our ground truth primitives (folder "prim\_gt") and the original ModelNet mesh (folder "ModelNet10\_mesh")

## Train
- For shape generation from scratch:
```
th driver.lua
```

- For shape generation conditioned on single depth map:
```
th driver_depth.lua
```

## Generation
- For shape generation from scratch:
```
th testNet_3dp.lua
```

- For shape generation conditioned on single depth map:
```
th testNet_3dp_depth.lua
```

## Visualization
- To visualize ground truth primitives, run visualizeGTPrimitive.m in Matlab
- To visualize sample shape generation, run visualizeRandomGeneration.m 
- To visualize sample shape generation conditioned on depth, run visualizeDepthReconGeneration.m

## Primitive ground truth
- See ./matlab/ folder

## Note
For shape generation conditioned on depth, as explained in the paper Sec 5.1,  we perform a nearest neighbor query based on the encoded feature of the depth map to retrieve the most similar shape in the training set and use the configuration as the initial state for the RNN. For convenience, we include our pre-computed initial configuration for each test class in folder "data/sample\_generation".

## Primitive parsing
We provide in the matlab folder the demo code (demo.m) to parse single primitive. To sequentially parse primitives in batch, see "script\_parse\_primitive.m". After each run of "script\_parse\_primitive.m", run "script\_parse\_primitive\_symmetry.m" to get the symmetry. With every three parses, try "script\_refine\_parse\_primitive.m" to refine the parsed primitives.

## Citation
```
@inproceedings{zou20173d,
  title={3d-prnn: Generating shape primitives with recurrent neural networks},
  author={Zou, Chuhang and Yumer, Ersin and Yang, Jimei and Ceylan, Duygu and Hoiem, Derek},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```

## Acknowledgement
- We express gratitudes to the torch implementation of [hand writting digits generation](https://github.com/jarmstrong2/handwritingnet) as we benefit from the code.
