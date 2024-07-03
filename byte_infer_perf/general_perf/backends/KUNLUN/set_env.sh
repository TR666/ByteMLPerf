#!/bin/bash
export TVM_DIR=`readlink -f .`
export XTCL_INSTALL_DIR=`readlink -f .`
export PYTHONPATH=$XTCL_INSTALL_DIR/python:$XTCL_INSTALL_DIR/python/tvm:$XTCL_INSTALL_DIR/python/topi:${PYTHONPATH}
export LD_LIBRARY_PATH=$XTCL_INSTALL_DIR/shlib:${LD_LIBRARY_PATH}
if [[ $XTCL_PARAM_CACHE -ne 1 ]] && [[ $XTCL_PARAM_CACHE_TEST -ne 1 ]];then
    export RT_LIBRARY_PATH=$XTCL_INSTALL_DIR/runtime/shlib
fi

export THIRDPARTY_LIB_DIR=`readlink -f 3rdparty/lib/`
export LD_LIBRARY_PATH=${THIRDPARTY_LIB_DIR}:${RT_LIBRARY_PATH}:${LD_LIBRARY_PATH}
export KERNEL_SEARCH_PATH=${XTCL_INSTALL_DIR}/xpu/kernels
export KERNEL_INCLUDE=${KERNEL_SEARCH_PATH}

export CLANG_PATH=${XTCL_INSTALL_DIR}

if [ $(ls /dev/xpu* | wc -l) -gt 1 ];then
    echo "##############    has card    ##############"
    echo "Current luanch on DEVICE_TYPE: $DEVICE_TYPE, real card"
    unset XPU_SIMULATOR_MODE
    if [ "$DEVICE_TYPE" == "KUNLUN2" ]; then
        export XPUSIM_DEVICE_MODEL=KUNLUN2
        #Set L3 size
        export XTCL_L3_SIZE=67104768
    elif [[ "$DEVICE_TYPE" == "KUNLUN3" ]]; then
        export XPUSIM_DEVICE_MODEL=KUNLUN3
        #Set L3 size
        export XTCL_L3_SIZE=100663296
    else
        export XPUSIM_DEVICE_MODEL=KUNLUN1
        #Set L3 size
        export XTCL_L3_SIZE=16776192
    fi
else
    echo "############## using simulator ##############"
    export XPU_SIMULATOR_MODE=1
    echo "Current luanch on DEVICE_TYPE: $DEVICE_TYPE, simulator"
    if [ "$DEVICE_TYPE" == "KUNLUN2" ]; then
        export XPUSIM_DEVICE_MODEL=KUNLUN2
        export XPUSIM_GM_BASE_ADDR=0x800000000
        export XPUSIM_L3_BASE_ADDR=0x0c0000000
        #Set L3 size
        export XTCL_L3_SIZE=67104768
    elif [[ "$DEVICE_TYPE" == "KUNLUN3" ]]; then
        export XPUSIM_DEVICE_MODEL=KUNLUN3
        export XPUSIM_SSE_LOG_LEVEL=DISABLE
        export XPUSIM_GM_BASE_ADDR=0x4000000000
        export XPUSIM_L3_BASE_ADDR=0x90000000
        #Set L3 size
        export XPUSIM_L3_SIZE=100663296
        export XTCL_L3_SIZE=100663296
    else
        export XPUSIM_DEVICE_MODEL=KUNLUN1
        #Set L3 size
        export XTCL_L3_SIZE=16776192
    fi
fi

# fast launch
if [[ "$DEVICE_TYPE" == "KUNLUN3" ]]; then
    export XPU_FORCE_USERMODE_LAUNCH=1
    export CUDART_DUMMY_REGISTER=1
    export XPU_DUMMY_EVENT=1
fi