# Breast metadata

Python module for loading and filtering metadata for Biomechanics for Breast 
Imaging studies. See the documentation for more information.

## Dependencies

See requirements.txt for python dependencies.
The python module named parameterized is used for testing.

## Accessing documentation
The documentation is hosted on an ABI webserver that is accessible only within
the university or via VPN - http://sitesdev.bioeng.auckland.ac.nz/psam012/breast-metadata/docs/build/index.html

## Building documentation locally

### From the terminal
1. Clone the repository to your local machine.

2. Navigate to the docs/

3. Issue the command: `make html`

4. Open the index.html file in `docs/build/` folder

### From Pycharm
See the following [instructions](https://bioengineering-toolbox.readthedocs.io/en/latest/documentation/sphinx.html#adding-a-sphinx-build-configuration-to-pycharm).

## Contributing to documentation

### Updating the documentation
1. Fork this repository to your github account.

2. Edit the restructuredText (.rst) or markdown (.md) files in the 
`docs/sources` folder (editing of these files can performed directly using the 
file editing tools provided by github. This will allow you to commit your 
changes to the repository.

3. Make a pull request from your fork to the master branch of tutorial repository in the abi-breast-biomechanics-group organisation.

4. Your changes will be reviewed and pulled into the main tutorial repository.

Over time, your fork may become out of sync with the master branch of this repository in the abi-breast-biomechanics-group organisation. Create a pull request on your fork to pull in changes from the master branch of this repository in the abi-breast-biomechanics-group to get your fork back up-to-date. This can be performed directly on the github website.

### Hosting the documentation
Any html files in `/people/www/psam012/` will be automatically hosted by ABI 
on `http://sitesdev.bioeng.auckland.ac.nz/psam012` for access only within the
university. Once the documentation has been built using `make html`, copy the 
`build` folder and paste it in 
`/people/wwww/psam012/breast-metadata/docs/build`. The documentation can then
be accessed from http://sitesdev.bioeng.auckland.ac.nz/psam012/breast-metadata/docs/build/index.html
