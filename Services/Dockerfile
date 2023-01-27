############################################################################
# Begin license text.
# Copyright Feb. 27, 2020 Robert A. Murphy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# End license text.
############################################################################

############################################################################
##
## File:      Dockerfile
##
## Purpose:   Recipe for the mtl.py docker file image creation
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

# requirements that need to be done
#
# create your own account <accountname> and repo <reponame> on docker hub
#
# install either docker desktop for windows or ubuntu
#
# docker run --name repo alpine/git clone https://github.com/robertauny:Azure-A-I.git
#
# docker cp repo:/git/Azure-A-I/ .
#
# cd Azure-A-I
#
# docker build -t <accountname>/<reponame>:<tag> . e.g. docker build -t robertauny/python-mtl:latest .
#
# docker run python-mtl

# version of python to use
FROM python:3.9 
# working directory where the python modules can be found
WORKDIR /Services
# make some needed directories
RUN mkdir -p ./models ./data/images ./data/csv
# copy the application files from the host to the docker working directory
COPY Services/*.py Services/*.json .
# copy the python module requirements from the host to the docker working directory
COPY pip_requirements.txt .
# install python modules
RUN pip install -r pip_requirements.txt
# execution of the python module in mtl.py
CMD python ./dataset.py
# copy the data file from the host to the docker working directory
COPY Services/data/csv/clicks.csv ./data/csv
# execution of the python module in mtl.py
CMD python ./mtl.py
