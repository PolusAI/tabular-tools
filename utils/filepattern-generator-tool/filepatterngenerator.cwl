class: CommandLineTool
cwlVersion: v1.2
inputs:
  chunkSize:
    inputBinding:
      prefix: --chunkSize
    type: double?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  groupBy:
    inputBinding:
      prefix: --groupBy
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
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
    dockerPull: polusai/filepattern-generator-tool:0.2.2-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
