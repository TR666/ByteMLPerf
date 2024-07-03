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

class RuntimeBackendKUNLUN(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(CompileBackendKUNLUN. self).__init__()
        self.hardware_type = 'KUNLUN'
        self.framework = None
        self.frozen_model_path = None
        self.need_reload = False
        self.model_runtimes = [] # {bs : runtime_module};
    def predict(self, feeds, pure_run=False):
        if not self.model_runtimes:
            log.error("No runtime_module!")
        results = {}
        local_check = {}

        if self.configs["is_dyn"]:
            m = self.model_runtimes[1]
            for i in range(len(self.configs["input_names"])):
                in_name = self.configs["input_names"][i]
                in_type = str.lower(self.configs["input_type"][i])
                set_input_start = time.time()
                if not isinstance(feeds[in_name+":0"], np.ndarray):
                    feeds[in_name+":0"] = np.array(feeds[in_name+":0"]).astype(in_type)
                m.set_dshape_input(in_name, feeds[in_name+":0"])
                set_input_end = time.time()
                set_input_elapsed_ms = 1000 * (set_input_end - set_input_start)
                local_check['set_input'] = [set_input_elapsed_ms]
        else:
            candite_bs = len(list(feeds.values())[0]) # all inputs has same batch_size
            origin_shapes = self.configs["segments"][0]["input_tensor_map"]
            bs = int(candite_bs / (list(origin_shapes.values())[0][0]))
            for i, key in enumerate(origin_shapes):
                if bs != int(len(list(feeds.values())[i]) / (origin_shapes[key][0])):
                    log.error("For input_name{}, we need input_shape[0] is {}, while feeds shape[0] is {}".format(key, bs*origin_shapes[key][0], candite_bs))
                #print("candite_bs is {}, assign bs is {}".format(candite_bs, bs))
            m = self.model_runtimes[bs]
            for i in range(len(self.configs["input_names"])):
                in_name = self.configs["input_names"][i]
                in_type = str.lower(self.configs["input_type"][i])
                set_input_start = time.time()
                if not isinstance(feeds[in_name+":0"], np.ndarray):
                        feeds[in_name+":0"] = np.array(feeds[in_name+":0"]).astype(in_type)
                m.set_input(in_name, feeds[in_name+":0"])
                set_input_end = time.time()
                set_input_elapsed_ms = 1000 * (set_input_end - set_input_start)
                local_check['set_input'] = [set_input_elapsed_ms]

        run_start = time.time()
        m.run()
        run_end = time.time()
        run_elapsed_ms = 1000 * (run_end - run_start)
        local_check['run'] = [run_elapsed_ms]

        get_out_start = time.time()
        num_out = m.get_num_outputs()
        for i in range(num_out):
            results[str(i)] = m.get_output(i).asnumpy()
        get_out_end = time.time()
        get_out_elapsed_ms = 1000 * (get_out_end - get_out_start)
        local_check['get_out'] = [get_out_elapsed_ms]

        if pure_run:
            results['t_run'] = run_elapsed_ms
            results['t_H2D'] = set_input_elapsed_ms
            results['t_D2H'] = get_out_elapsed_ms

        log.debug("local check time is: {}".format(local_check))

        return results

    def get_tail_latency(self, reslut_list):
        t_tail_latency = round(reslut_list[int(len(reslut_list) * 0.99)], 2)
        return t_tail_latency

    def benchmark(self):
        batch_sizes = self.workload['batch_sizes']
        iterations = self.workload['iterations']
        reports = []
        for batch_size in batch_sizes:
            times_range = []
            pure_run_times_range = {}
            pure_run_times_range['t_H2D'] = []
            pure_run_times_range['t_run'] = []
            pure_run_times_range['t_D2H'] = []

            report = {}
            report['BS'] = batch_size
            test_data = self._get_fake_samples(
                batch_size, self.configs['segments'][0]['input_tensor_map'], self.configs['input_type'])

            # ======================== all run out ========================
            for _ in range(30):
                self.predict(test_data, pure_run=True)

            for _ in range(iterations):
                start_time = time.time()
                result = self.predict(test_data, pure_run=True)
                end_time = time.time()
                times_range.append(end_time - start_time)
                pure_run_times_range['t_H2D'].append(result['t_H2D'])
                pure_run_times_range['t_run'].append(result['t_run'])
                pure_run_times_range['t_D2H'].append(result['t_D2H'])

            times_range.sort()
            tail_latency = round(
                times_range[int(len(times_range) * 0.99)] * 1000, 2)
            avg_latency = round(sum(times_range) / iterations * 1000, 2)
            qps = int(1000.0 * batch_size / avg_latency)

            pure_run_times_range['t_run'].sort()
            pure_run_times_range['t_H2D'].sort()
            pure_run_times_range['t_D2H'].sort()
            pure_run_tail_latency = round(
                pure_run_times_range['t_run'][int(len(pure_run_times_range['t_run']) * 0.99)], 2)
            pure_run_avg_latency = round(sum(pure_run_times_range['t_run']) / iterations, 2)
            pure_run_qps = int(1000.0 * batch_size / pure_run_avg_latency)
            log.info('Batch size is {}, QPS: {}, Avg Latency:{}, '
                    'pure run QPS: {}, pure run Avg Latency:{}, pure run Tail Latency:{}'.format(
                batch_size, qps, avg_latency, pure_run_qps, pure_run_avg_latency, pure_run_tail_latency))

            report['QPS'] = qps
            report['AVG Latency'] = avg_latency
            report['P99 Latency'] = tail_latency
            report['pure run         QPS'] = pure_run_qps
            report['pure run AVG Latency'] = pure_run_avg_latency
            report['pure run P99 Latency'] = pure_run_tail_latency
            report['pure H2D P99 Latency'] = self.get_tail_latency(pure_run_times_range['t_H2D'])
            report['pure D2H P99 Latency'] = self.get_tail_latency(pure_run_times_range['t_D2H'])

            reports.append(report)

        return reports

    def _get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if (input_type == "int64"):
            input_type = "int32"
        if input_type:
            i = 0
            for key, val in shape.items():
                val = [val[0] * batch_size] + val[1:]
                data[key] = ((np.random.random(size=val) + 1)).astype(
                    INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")
