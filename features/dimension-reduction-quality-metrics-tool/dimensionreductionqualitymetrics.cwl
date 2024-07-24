class: CommandLineTool
cwlVersion: v1.2
inputs:
  originalDir:
    inputBinding:
      prefix: --originalDir
    type: Directory
  originalPattern:
    inputBinding:
      prefix: --originalPattern
    type: string?
  embeddedDir:
    inputBinding:
      prefix: --embeddedDir
    type: Directory
  embeddedPattern:
    inputBinding:
      prefix: --embeddedPattern
    type: string?
  numQueries:
    inputBinding:
      prefix: --numQueries
    type: int?
  ks:
    inputBinding:
      prefix: --ks
    type: string?
  distanceMetrics:
    inputBinding:
      prefix: --distanceMetrics
    type: string?
  qualityMetrics:
    inputBinding:
      prefix: --qualityMetrics
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/dimension-reduction-quality-metrics-tool:0.1.0-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
