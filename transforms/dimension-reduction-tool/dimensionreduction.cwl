class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  algorithm:
    inputBinding:
      prefix: --algorithm
    type: string?
  nComponents:
    inputBinding:
      prefix: --nComponents
    type: int
  pcaWhiten:
    inputBinding:
      prefix: --pcaWhiten
    type: boolean?
  pcaSvdSolver:
    inputBinding:
      prefix: --pcaSvdSolver
    type: string?
  pcaTol:
    inputBinding:
      prefix: --pcaTol
    type: float?
  tsnePerplexity:
    inputBinding:
      prefix: --tsnePerplexity
    type: float?
  tsneEarlyExaggeration:
    inputBinding:
      prefix: --tsneEarlyExaggeration
    type: float?
  tsneLearningRate:
    inputBinding:
      prefix: --tsneLearningRate
    type: float?
  tsneMaxIter:
    inputBinding:
      prefix: --tsneMaxIter
    type: int?
  tsneMetric:
    inputBinding:
      prefix: --tsneMetric
    type: string?
  tsneInitNComponents:
    inputBinding:
      prefix: --tsneInitNComponents
    type: int?
  umapNNeighbors:
    inputBinding:
      prefix: --umapNNeighbors
    type: int?
  umapNEpochs:
    inputBinding:
      prefix: --umapNEpochs
    type: int?
  umapMinDist:
    inputBinding:
      prefix: --umapMinDist
    type: float?
  umapSpread:
    inputBinding:
      prefix: --umapSpread
    type: float?
  umapMetric:
    inputBinding:
      prefix: --umapMetric
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/dimension-reduction-tool:0.1.0-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
