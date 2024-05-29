class: CommandLineTool
cwlVersion: v1.2
inputs:
  batchSize:
    inputBinding:
      prefix: --batchSize
    type: string?
  catEmbDim:
    inputBinding:
      prefix: --catEmbDim
    type: string?
  classifier:
    inputBinding:
      prefix: --classifier
    type: string
  clipValue:
    inputBinding:
      prefix: --clipValue
    type: double?
  computeImportance:
    inputBinding:
      prefix: --computeImportance
    type: boolean?
  deviceName:
    inputBinding:
      prefix: --deviceName
    type: string
  dropLast:
    inputBinding:
      prefix: --dropLast
    type: boolean?
  epsilon:
    inputBinding:
      prefix: --epsilon
    type: double?
  evalMetric:
    inputBinding:
      prefix: --evalMetric
    type: string
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  gamma:
    inputBinding:
      prefix: --gamma
    type: double?
  groupedFeatures:
    inputBinding:
      prefix: --groupedFeatures
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  lambdaSparse:
    inputBinding:
      prefix: --lambdaSparse
    type: double?
  lossFn:
    inputBinding:
      prefix: --lossFn
    type: string
  lr:
    inputBinding:
      prefix: --lr
    type: double?
  maskType:
    inputBinding:
      prefix: --maskType
    type: string
  maxEpochs:
    inputBinding:
      prefix: --maxEpochs
    type: string?
  momentum:
    inputBinding:
      prefix: --momentum
    type: double?
  nA:
    inputBinding:
      prefix: --nA
    type: string?
  nD:
    inputBinding:
      prefix: --nD
    type: string?
  nIndepDecoder:
    inputBinding:
      prefix: --nIndepDecoder
    type: string?
  nIndependent:
    inputBinding:
      prefix: --nIndependent
    type: string?
  nShared:
    inputBinding:
      prefix: --nShared
    type: string?
  nSharedDecoder:
    inputBinding:
      prefix: --nSharedDecoder
    type: string?
  nSteps:
    inputBinding:
      prefix: --nSteps
    type: string?
  numWorkers:
    inputBinding:
      prefix: --numWorkers
    type: string?
  optimizerFn:
    inputBinding:
      prefix: --optimizerFn
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  patience:
    inputBinding:
      prefix: --patience
    type: string?
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  schedulerFn:
    inputBinding:
      prefix: --schedulerFn
    type: string
  seed:
    inputBinding:
      prefix: --seed
    type: string?
  stepSize:
    inputBinding:
      prefix: --stepSize
    type: string?
  targetVar:
    inputBinding:
      prefix: --targetVar
    type: string
  testSize:
    inputBinding:
      prefix: --testSize
    type: double?
  virtualBatchSize:
    inputBinding:
      prefix: --virtualBatchSize
    type: string?
  warmStart:
    inputBinding:
      prefix: --warmStart
    type: boolean?
  weights:
    inputBinding:
      prefix: --weights
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/pytorch-tabnet-tool:0.1.0-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
