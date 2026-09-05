import onnxruntime


class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", awd=False):
        self.onnx_model_path = onnx_model_path
        # The policy is a ~900 KB MLP that infers in 0.03 ms, but the default
        # session grabs one thread per core and those threads spin-wait between
        # calls. At 50 Hz that leaves a whole CPU busy doing nothing - on a
        # 16-core laptop it pegged every core and made the machine unusable.
        # One thread, no spinning: same latency, idle CPU between steps.
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = input_name
        self.awd = awd

    def infer(self, inputs):
        if self.awd:
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            return outputs[0][0]
        else:
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            return outputs[0]


if __name__ == "__main__":
    import argparse
    import numpy as np
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    args = parser.parse_args()

    obs_size = 46

    oi = OnnxInfer(args.onnx_model_path, awd=True)
    times = []
    for i in range(1000):
        inputs = np.random.uniform(size=obs_size).astype(np.float32)
        # inputs = np.arange(obs_size).astype(np.float32)
        # print(inputs)
        start = time.time()
        print(oi.infer(inputs))
        times.append(time.time() - start)

    print("Average time: ", sum(times) / len(times))
    print("Average fps: ", 1 / (sum(times) / len(times)))
