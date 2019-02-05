ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION} as base

MAINTAINER Francesco Ceccon <francesco@ceccon.me>


ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
	build-essential \
	gfortran \
	wget \
	git \
	unzip \
	python3 \
	python3-pip \
	libblas-dev \
	liblapack-dev \
	cppad \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade \
	pip setuptools numpy


# Install pybind11, used to bind GALINI with c++ libraries
ARG PYBIND11_COMMIT_SHA=64205140bdaf02be50d3476bb507e8354a512d04
RUN mkdir /pybind11 \
	&& wget -O /pybind11/pybind11.zip "https://github.com/pybind/pybind11/archive/${PYBIND11_COMMIT_SHA}.zip"\
	&& unzip -d /pybind11 /pybind11/pybind11.zip \
	&& rm -f /pybind11/pybind11.zip
ENV PYBIND11_INCLUDE_DIR=/pybind11/pybind11-${PYBIND11_COMMIT_SHA}/include

# Install Ipopt
ARG IPOPT_VERSION=3.12.12
RUN mkdir /ipopt \
	&& wget -O /ipopt/ipopt.zip "https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.zip" \
	&& unzip -d /ipopt /ipopt/ipopt.zip \
	&& cd /ipopt/Ipopt-${IPOPT_VERSION} \
	&& cd ThirdParty/Metis && ./get.Metis && cd ../.. \
	&& cd ThirdParty/Mumps && ./get.Mumps && cd ../.. \
	&& ./configure --with-blas="-lblas -llapack" --with-lapack="-llapack" --prefix="/ipopt" \
	&& make install \
	&& cd ../ && rm -rf Ipopt-${IPOPT_VERSION} ipopt.zip

ENV IPOPT_INCLUDE_DIR=/ipopt/include
ENV IPOPT_LIBRARY_DIR=/ipopt/lib
ENV LD_LIBRARY_PATH=/ipopt/lib

RUN mkdir /galini/
COPY setup.py /galini/
COPY setup.cfg /galini/
COPY src /galini/src
COPY galini /galini/galini
COPY tests /galini/tests
COPY LICENSE /galini/
COPY *.rst /galini/

RUN cd /galini \
	&& python3 setup.py build \
	&& python3 setup.py install

RUN galini -h
