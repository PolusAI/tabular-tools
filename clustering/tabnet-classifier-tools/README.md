# PyTorch TabNet tool(v0.1.0-dev0)

This tool uses [tabnet](https://github.com/dreamquark-ai/tabnet/tree/develop), a deep learning model designed for tabular data structured in rows and columns. TabNet is suitable for classification, regression, and multi-task learning.

## Inputs:

### Input data:
The input tabular data that need to be trained. This plugin supports `.csv`, `.feather`and `.arrow` file formats

### Details:

PyTorch-TabNet can be employed for:.

1. TabNetClassifier: For binary and multi-class classification problems
2. TabNetRegressor: For simple and multi-task regression problems
3. TabNetMultiTaskClassifier: multi-task multi-classification problems


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes 38 input arguments and one output argument:

| Name             | Description                                                                 | I/O    | Type          |
| ---------------- | --------------------------------------------------------------------------- | ------ | ------------- |
| `--inpdir`       | Input tabular data                                                          | Input  | genericData   |
| `--filePattern`  | Pattern to parse tabular files                                              | Input  | string        |
| `--testSize`     | Proportion of the dataset to include in the test set                        | Input  | number        |
| `--nD`           | Width of the decision prediction layer                          | Input  | integer       |
| `--nA`           | Width of the attention embedding for each mask                          | Input  | integer       |
| `--nSteps`       | Number of steps in the architecture                         | Input  | integer       |
| `--gamma`        | Coefficient for feature reuse in the masks                         | Input  | number       |
| `--catEmbDim`    | List of embedding sizes for each categorical feature                          | Input  | integer       |
| `--nIndependent`    | Number of independent Gated Linear Unit layers at each step          | Input  | integer       |
| `--nShared`    | Number of shared Gated Linear Unit layers at each step          | Input  | integer       |
| `--epsilon`     | Constant value                        | Input  | number        |
| `--seed`    | Random seed for reproducibility          | Input  | integer       |
| `--momentum`     | Momentum for batch normalization                        | Input  | number        |
| `--clipValue`     | Clipping of the gradient value                        | Input  | number        |
| `--lambdaSparse`     | Extra sparsity loss coefficient                        | Input  | number        |
| `--optimizerFn`     | Pytorch optimizer function                        | Input  | enum        |
| `--lr`     | learning rate for the optimizer                        | Input  | number         |
| `--schedulerFn`     | Parameters used initialize the optimizer                        | Input  | enum         |
| `--stepSize`     | Parameter to apply to the scheduler_fn                        | Input  | integer         |
| `--deviceName`     | Platform used for training                        | Input  | enum         |
| `--maskType`     | A masking function for feature selection                        | Input  | enum         |
| `--groupedFeatures`     | Allow the model to share attention across features within the same group | Input  | integer         |
| `--nSharedDecoder`     | Number of shared GLU block in decoder | Input  | integer         |
| `--nIndepDecoder`     | Number of independent GLU block in decoder | Input  | integer         |
| `--evalMetric`     | Metrics utilized for early stopping evaluation                        | Input  | enum         |
| `--maxEpochs`     | Maximum number of epochs for training | Input  | integer         |
| `--patience`     | Consecutive epochs without improvement before early stopping | Input  | integer         |
| `--weights`     | Sampling parameter only for TabNetClassifier | Input  | integer         |
| `--lossFn`     | Loss function                        | Input  | enum         |
| `--batchSize`     | Batch size                       | Input  | integer         |
| `--virtualBatchSize`     | Size of mini-batches for Ghost Batch Normalization    | Input  | integer         |
| `--numWorkers`     | Number or workers used in torch.utils.data.Dataloader    | Input  | integer         |
| `--dropLast`     | Option to drop incomplete last batch during training    | Input  | boolean         |
| `--warmStart`     | For scikit-learn compatibility, enabling fitting the same model twice    | Input  | boolean         |
| `--targetVar`     | Target feature containing classification labels   | Input  | string         |
| `--computeImportance`     | Compute feature importance    | Input  | boolean         |
| `--classifier`     | Pytorch tabnet Classifier for training   | Input  | enum         |
| `--preview`      | Generate JSON file of sample outputs                       | Input | boolean         |
| `--outdir`       | Output collection                            | Output | genericData   |
