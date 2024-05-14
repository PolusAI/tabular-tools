#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rt-cetsa-analysis-simple-tool:"${version}"
