import os
import json
import logging
import torch
import time
import numpy as np

from general_perf.backends import compile_backend

log = logging.getLogger("BackendKUNLUN")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "LONG": torch.long
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64
}

# def _init_vsl_env():
#     log.info("[DEBUG]: ENABLE var-len for bert")
#     os.environ['XTCL_ENABLE_VSL'] = '1'

class CompileBackendKUNLUN(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendKUNLUN, self).__init__()
        self.hardware_type = 'KUNLUN'
        self.framework = None
        self.frozen_model_path = None
        self.need_reload = False
        self.model_runtimes = [] # {bs : runtime_module};
    
    def pre_optimize(self, configs: Dict[str, Any]):
        """
        Model pre-optimization interface. Requirements: Model pre-optimization
        cannot change the model format. Torch model export to ONNX is allowed.
        """
        self._update_model_config(configs.get("interact_info", {}))

        self.precision = (
            self.model_config.get("converter_options", {})
            .get("precision", "FP32")
            .upper()
        )
        if self.model_config.get("pack_config"):
            self.packrunner = True

        model_info = configs["model_info"]
        model_type = model_info["model_format"]
        model_name = model_info["model"]

        pre_optimized_root = Path(self.current_dir) / "pre_optimized_models"
        if not pre_optimized_root.exists():
            pre_optimized_root.mkdir(parents=True)

        model_path = os.path.abspath(configs["model_info"]["model_path"])
        onnx_path = pre_optimized_root / (model_name + ".onnx")

        if not self.model_config:
            self.model_config = configs.get("interact_info", {})

        # convert model to onnx if it's not
        # configs['workload'] is the content of workloads/<task_name>.json and
        # configs['model_info'] is content of model_zoo/<task_name>.json
        if model_type != "onnx":
            if onnx_path.exists():
                onnx_path = self._update_pack_model(onnx_path, model_info)
                model_info["model_path"] = onnx_path
                log.info("{} file exists, skip ONNX conversion".format(onnx_path.name))
            else:
                # convert the model to onnx
                log.info(
                    "Convert the model: {} from format: {} to onnx".format(
                        model_name, model_type
                    )
                )
                if model_type == "saved_model":
                    saved_to_onnx.savedmodel_to_onnx(model_path, onnx_path)
                    onnx_path = self._update_pack_model(onnx_path, model_info)
                elif model_type == "pt":
                    torch_to_onnx.torch_to_onnx(model_path, str(onnx_path))
                    onnx_path = self._update_pack_model(onnx_path, model_info)
                else:
                    log.error(
                        "Wrong model type: {}, which must be saved_model, pt, or onnx".format(
                            model_type
                        )
                    )
                    raise TypeError("Model type must be saved_model, pt, or onnx")

                if os.path.exists(onnx_path):
                    model_info["model_path"] = onnx_path
                    log.info(
                        "Converted the model: {} from format: {} to onnx".format(
                            model_name, model_type
                        )
                    )
                else:
                    log.error(
                        "{} not exists, failed to convert the model: {} to onnx".format(
                            onnx_path, model_name
                        )
                    )
                    raise RuntimeError("Failed to convert model to onnx")
        else:
            log.info("{} is onnx model, skip ONNX conversion".format(model_name))

        return configs

    def _update_model_config(self, interact_info):
        # update poprt configuration based on interact_info
        if not self.model_config:
            self.model_config = {}
            self.model_config["converter_options"] = interact_info.get(
                "converter_options", {}
            )
            self.model_config["clients"] = int(interact_info.get("clients", "1"))
            batch_sizes = interact_info.get("batch_sizes", "").split(",").remove("")
            if batch_sizes:
                self.model_config["batch_sizes"] = [int(x.strip()) for x in batch_sizes]
            self.model_config["compiler_options"] = json.loads(
                interact_info.get("compiler_options", "{}")
            )

            self.model_config["clients"] = int(self.model_config.get("clients", "1"))
            batch_sizes = self.model_config.get("batch_sizes", "").split(",")
            if batch_sizes:
                self.model_config["batch_sizes"] = [
                    int(x.strip()) for x in batch_sizes if x.strip().isdigit()
                ]
            for key, value in self.model_config.items():
                if "_options" in key and isinstance(value, str):
                    self.model_config[key] = json.loads(value)

        if interact_info.get("precision"):
            self.model_config["converter_options"]["precision"] = interact_info[
                "precision"
            ]
            # update converter config when user selected fp8 in interact sections
            # and there is fp8_configs in interact_info config file
            if interact_info["precision"] == "fp8" and self.model_config.get(
                "fp8_configs"
            ):
                for config_name, config_section in self.model_config[
                    "fp8_configs"
                ].items():
                    if isinstance(self.model_config[config_name], dict):
                        self.model_config[config_name].update(config_section)
                    else:
                        self.model_config[config_name] = config_section

                del self.model_config["fp8_configs"]
    # bref: assert shape_dict[key][0] == batch_size
    def _build_dyn(self, shape_dict, output_nodes, layout="NCHW"):
        # build_dyn : whatever the batch_size is

        target = 'xpu -libs=xdnn -split-device-funcs -device-type=xpu2 -is_xpu_build_dyn_shape'
        target_host = 'llvm'
        device_id = 0
        ctx = tvm.device(target, device_id)
        # TF 2.4.0
        with tf.io.gfile.GFile(self.configs["frozen_model_path"], 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
            with tf.compat.v1.Session() as sess:
                graph_def = tf_testing.AddShapesToGraphDef(sess, output_nodes)

        mod, params = relay.frontend.from_tensorflow(graph_def, layout="NCHW",
                                      shape=shape_dict, outputs=output_nodes)

        log.debug("Build Begin ........")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod,
                                            target=target,
                                            target_host=target_host,
                                            params=params)
        log.debug("Build End ........")

        log.debug("Create Runtime Module ........")
        m = graph_executor.create(graph, lib, ctx)
        m.set_input(**params)
        log.debug("Create Runtime Module End........")

        self.model_runtimes[1] = m
        log.info("build dynamic module success")
        return True

    # build static module
    def _build_stc(self, bs_list, shape_dict, output_nodes, layout="NCHW"):
        target = 'xpu -libs=xdnn -split-device-funcs -device-type=xpu2'
        target_host = 'llvm'
        device_id = 0
        ctx = tvm.device(target, device_id)
        # TF 2.4.0
        with tf.io.gfile.GFile(self.configs["frozen_model_path"], 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
            with tf.compat.v1.Session() as sess:
                graph_def = tf_testing.AddShapesToGraphDef(sess, output_nodes)

        for bs in bs_list:
            if int(bs) in self.model_runtimes:
                continue
            new_shape_dict = {}
            for key in shape_dict:
                shape = shape_dict[key]
                new_shape = shape.copy()
                new_shape[0] = int(bs) * shape[0]
                new_shape_dict[key] = new_shape

            mod, params = relay.frontend.from_tensorflow(graph_def, layout="NCHW", shape=new_shape_dict, outputs=output_nodes)

            log.debug("Batch Size = {}, Build Begin ........".format(bs))
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod,
                                                target=target,
                                                target_host=target_host,
                                                params=params)
            log.debug("Batch Size = {}, Build End ........".format(bs))

            log.debug("Batch Size = {}, Create Runtime Module ........".format(bs))
            m = graph_executor.create(graph, lib, ctx)
            m.set_input(**params)
            log.debug("Batch Size = {}, Create Runtime Module End........".format(bs))

            self.model_runtimes[int(bs)] = m
        log.info("build static module for batch size {} success.".format(bs_list))
        return True

    def compile(self, config, dataloader=None):

        if self.framework == "Tensorflow" and (not os.path.exists(self.frozen_model_path)):
            log.error('Frozen model path: {} does not exist, please check.'.format(self.frozen_model_path))

        # init vsl env if set dataset_name == squad_vsl
        # if ("squad_vsl" in config['model_info']['dataset_name']):
        #     _init_vsl_env()
        
        result = {
            "model": config['model_info']['model'],
            "framework": config['model_info']['framework'],
            "compile_precision": 'INT16',
            "input_type": config['model_info']['input_type'].split(","),
            "max_batch_size": config['model_info']['max_batch_size'],
            "batch_sizes": config['workload']['batch_sizes'],
            "frozen_model_path": self.frozen_model_path,
            "dataset_name": config['model_info']['dataset_name'],
            "sg_percent": 100,
            "segments": [{
                "sg_idx": 0,
                "is_fallback": False,
                "input_tensor_map": config['model_info']['input_shape'],
                "output_tensor_map": config['model_info']['outputs'],
                "compiled_model": [
                    {
                        "compiled_bs": 1,
                        "compiled_obj": config['model_info']['model_path'],
                    },
                ],
            }, ]
        }

        self.workload = config['workload']
        log.debug("config.all is {}".format(config))
        log.debug("config.workload is {}".format(self.workload))

        # format in/o names
        model_name = config['model_info']['model']
        input_names = [str(var.split(":")[0]).strip() for var in result["segments"][0]["input_tensor_map"]]
        input_shapes = [result["segments"][0]["input_tensor_map"][var] for var in result["segments"][0]["input_tensor_map"]]
        shape_dict = {}
        for t_name, shape in zip(input_names, input_shapes):
            shape_dict[t_name] = shape
        output_names = [result["segments"][0]["output_tensor_map"].split(":")[0]]
        log.debug("input_names={}, input_shapes={}, output_names={}".format(input_names, input_shapes, output_names))

        result["output_names"] = output_names
        result["input_names"] = input_names

        if ('resnet' in model_name):
            result["is_dyn"] = True
        else:
            result["is_dyn"] = False

        self.configs = result
        if result["is_dyn"]:
            ret = self._build_dyn(shape_dict, result["output_names"], layout="NCHW")
            if ret:
                self.configs["compile_status"] = "success"
        else:
            ret = self._build_stc(config['workload']['batch_sizes'], shape_dict, result["output_names"], layout="NCHW")
            if ret:
                self.configs["compile_status"] = "success"
            compile_info = {
                    "compiled_bs": 1,
                    "compiled_obj": config['model_info']['model_path'],
                }
            for bs in config['workload']['batch_sizes'][1:]:
                compile_info["compiled_bs"] = int(bs)
                self.configs["segments"][0]["compiled_model"].append(compile_info.copy())

        log.debug("self configs is {}".format(result))
        return result

    def get_interact_profile(self, config):
        model_profile = []
        file_path = "general_perf/backends/KUNLUN/" + self.hardware_type + '.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                model_profile = json.load(f)
        else:
            log.warning('Interact profile: {} does not exist, if you need the interact_profile please check'.format(file_path))
        return model_profile

# TODO:是否需要提供best_batch
    def get_best_batch_size(self):
        """
        Get Best Batch Size for the model
        """
        return None
    