#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rt-cetsa-analysis-tool:"${version}"
