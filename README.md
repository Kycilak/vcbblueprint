# Visual Circuit Board (VCB) Python Library

## Description

This Python library provides a way to read Visual Circuit Board (VCB) blueprint strings and convert them into a convenient data structure.

## Installation

    pip install vcbblueprint

## Usage

Here is a simple usage example:

```python
import vcbblueprint

blueprint_string = "VCB+..." # a VCB blueprint string

# read the blueprint string and get the version and layers
version, layers = vcbblueprint.read_blueprint(blueprint_string)

# use the filter function to get a boolean matrix of pixels of a
# specific component type
and_pixels = vcbblueprint.filter(layers[0], vcbblueprint.ComponentType.AND)
```

## Contributions

Contributions are welcome. Feel free to open an issue or submit a pull request.
