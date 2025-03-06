# BASE_TORCH_TAG=${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV TORCH_VERSION=2.0.1
ENV CUDA_VERSION=11.7
ENV CUDNN_VERSION=8

# RUN export BUILDKIT_PROGRESS=plain


# Fix nvidia-key error issue (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/*.list

# RUN --mount=type=cache,target=/opt/conda/pkgs \
#     conda install -n base -c conda-forge mamba -y && \
#     mamba clean -ya

# Installing apt packages
RUN export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	  git wget tmux vim zsh build-essential cmake ninja-build libopenblas-dev libsparsehash-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# RUN conda install -n base -c conda-forge mamba -y && \
#     mamba clean -ya

# COPY environment.yml .
# RUN mamba env create -f environment.yml && \
#     mamba clean -ya

COPY environment.yml .
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
# RUN --mount=type=cache,target=/opt/conda/pkgs \
#     mamba env create -f environment.yml && \
#     mamba clean -ya
RUN conda env create -f environment.yml 


# Install Pointcept environment
# RUN conda install h5py=3.13.0 pyyaml=6.0.2 -c pyg -c pytorch -c nvidia/label/cuda-12.1.1 -c nvidia -c bioconda -c conda-forge -c defaults -y 
# RUN conda install tensorboard=2.19.0 tensorboardx=2.6.2.2 yapf=0.43.0 addict=2.4.0 einops=0.8.1 scipy=1.15.2 plyfile=1.1 termcolor=2.5.0 timm=1.0.15-c pyg -c pytorch -c nvidia/label/cuda-12.1.1 -c nvidia -c bioconda -c conda-forge -c defaults -y
# RUN conda install pytorch-cluster=1.6.3 pytorch-scatter=2.1.2 pytorch-sparse=0.6.18 -c pyg -c pytorch -c nvidia/label/cuda-12.1.1 -c nvidia -c bioconda -c conda-forge -c defaults -y

# RUN pip install --upgrade pip
# RUN pip install torch-geometric==2.6.1
# RUN pip install spconv-cu118==2.3.8
# RUN pip install open3d

# # Build MinkowskiEngine
# RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
# WORKDIR /workspace/MinkowskiEngine
# RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" python setup.py install --blas=openblas --force_cuda
# WORKDIR /workspace

# # Build pointops
# RUN git clone https://github.com/Pointcept/Pointcept.git
# RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointops -v

# # Build pointgroup_ops
# RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointgroup_ops -v

# # Build swin3d
# RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0" pip install -U git+https://github.com/microsoft/Swin3D.git -v


python tools/test.py --config-file configs/scannet/semseg-pt-v3m1-0-base.py  \
       --options weight=PointTransformerV3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth
