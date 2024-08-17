#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rt-cetsa-moltenprot-tool:"${version}"
