class: CommandLineTool
cwlVersion: v1.2
baseCommand: ["python3", "-m", "polus.tabular.regression.rt_cetsa_analysis"]
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  params:
    inputBinding:
      prefix: --params
    type: string
  values:
    inputBinding:
      prefix: --values
    type: string
  platemap:
    inputBinding:
      prefix: --platemap
    type: File
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  EnvVarRequirement:
    envDef:
      WORKDIR: /opt/executables/
  DockerRequirement:
    dockerPull: polusai/rt-cetsa-analysis-simple-tool:0.1.0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
