========
Metadata
========

This section of the documentation outlines the metadata that accompanies study
data.

Style guide
===========

* Variables are underscore_lowercase
* The analysis type needs to be prepended to each metadata_key e.g.
  `geometry_ef`.

Metadata schema
===============

The metadata schema describing the metadata keys are documented as json files
in following files:

* `docs/source/study_metadata_fields.json`
* `docs/source/image_metadata_fields.json`
* `docs/source/geometry_metadata_fields.json`
* `docs/source/pressure_metadata_fields.json`

The contents of these files are displayed in the following sections.

.. jsonschema:: schema.json


.. jsonschema:: language.schema.json

Used types:
.. jsonschema:: common.schema.json#/definitions/languageTag

.. jsonschema:: common.schema.json#/definitions/localizedText

.. jsonschema:: common.schema.json#/definitions/rodCode

.. jsonschema:: common.schema.json#/definitions/trimmedText