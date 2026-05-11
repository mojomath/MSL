#!/bin/bash
set -e

for f in tests/*.mojo; do
    pixi run mojo run -I ./ -I tests/ "$f"
done
