"""Python 3.6 numeric-only TF1 worker. Never imports TimeSigma or EasyTSAD."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def array_binding(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest(value.tobytes())}


def initialize(request):
    if set(request) != {"action", "directory", "source_root", "sources", "configuration", "seed", "reference"}:
        raise ValueError("invalid initialization fields")
    if request["seed"] != 1 or type(request["seed"]) is not int:
        raise ValueError("worker requires canonical seed=1")
    source_root = Path(request["source_root"])
    for source in request["sources"]:
        directory = source_root / source["name"]
        for name, expected in source["files"].items():
            if digest((directory / name).read_bytes()) != expected:
                raise ValueError("worker source mismatch: " + name)
        sys.path.insert(0, str(directory))
    import pkg_resources
    versions = {}
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines():
        name, expected = line.split("==")
        actual = pkg_resources.get_distribution(name).version
        if actual != expected:
            raise ValueError("worker dependency mismatch: " + name)
        versions[name] = actual
    if platform.python_version() != "3.6.13" or platform.machine() != "x86_64":
        raise ValueError("worker requires the qualified Python 3.6.13 x86_64 runtime")
    runtime = {"python": platform.python_version(), "machine": platform.machine(),
               "libc": list(platform.libc_ver()), "packages": versions,
               "device": "cpu", "tf_intra_threads": 1, "tf_inter_threads": 1,
               "omp_threads": 1, "blas_threads": 1, "inter_op_thread_pool": -1}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("worker requires single-thread numeric libraries")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("worker requires CPU execution")
    return Worker(request, runtime)


class Worker:
    def __init__(self, request, runtime):
        import numpy as np
        import tensorflow as tf
        self.np, self.tf = np, tf
        self.root = Path(request["directory"])
        self.configuration = request["configuration"]
        self.config = SimpleNamespace(**self.configuration["Model_Params"])
        self.runtime = runtime
        self.reference = request["reference"]
        self.records, self.noises, self.occurrences = [], {}, []
        self.session = None
        self.trained = False
        self.replaying = False
        np.random.seed(1)
        tf.set_random_seed(1)

    def read_array(self, record):
        if set(record) != {"path", "shape", "dtype", "sha256"}:
            raise ValueError("numeric input binding fields differ")
        path = self.root / record["path"]
        if path.resolve().parent != self.root.resolve():
            raise ValueError("numeric input path must be worker-local")
        values = self.np.load(str(path), allow_pickle=False)
        if (values.ndim != 2 or values.shape[1] != 1 or values.dtype != self.np.float32
                or not len(values) or not self.np.isfinite(values).all()
                or array_binding(values) != {k: record[k] for k in ("shape", "dtype", "sha256")}):
            raise ValueError("invalid finite float32 [N,1] numeric input or binding")
        return values

    def session_open(self):
        tf = self.tf

        class SynchronousSession(tf.Session):
            def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
                # TF1.12's one-thread pool can still interleave caller/pool
                # execution and alter floating gradient accumulation. Its
                # native -1 policy executes our single CPU executor in the
                # calling thread; the model graph and objectives are unchanged.
                policy = tf.RunOptions()
                if options is not None:
                    policy.CopyFrom(options)
                if policy.inter_op_thread_pool not in (0, -1):
                    raise ValueError("OmniAnomaly requires synchronous CPU execution")
                policy.inter_op_thread_pool = -1
                return super(SynchronousSession, self).run(
                    fetches, feed_dict=feed_dict, options=policy, run_metadata=run_metadata)

        self.session = SynchronousSession(config=tf.ConfigProto(device_count={"GPU": 0},
                                           intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        self.session.__enter__()

    def build(self):
        from omni_anomaly.model import OmniAnomaly
        from omni_anomaly.training import Trainer
        from omni_anomaly.prediction import Predictor
        from tfsnippet.utils import get_variables_as_dict
        tf, c = self.tf, self.config
        with tf.variable_scope("model") as scope:
            model = OmniAnomaly(c, name="model")
            self.trainer = Trainer(model, model_vs=scope, max_epoch=c.max_epoch,
                                   batch_size=c.batch_size, valid_batch_size=c.test_batch_size,
                                   initial_lr=c.initial_lr, lr_anneal_epochs=c.lr_anneal_epoch_freq,
                                   lr_anneal_factor=c.lr_anneal_factor, grad_clip_norm=c.gradient_clip_norm,
                                   valid_step_freq=c.valid_step_freq)
            self.predictor = Predictor(model, batch_size=c.batch_size, n_z=c.test_n_z, last_point_only=True)
        self.variables = get_variables_as_dict(scope)
        self.model_variables = self.trainer._train_params
        self.session_open()

    def state(self):
        return {name: digest(value.tobytes()) for name, value in
                zip(self.model_variables, self.session.run(list(self.model_variables.values())))}

    def train(self, request):
        if self.session is not None or set(request) != {"action", "train", "valid"}:
            raise ValueError("fresh worker and numeric train/valid required")
        train, valid = self.read_array(request["train"]), self.read_array(request["valid"])
        np, c = self.np, self.config
        values = np.concatenate((train, valid), axis=0)
        n = int(len(values) * .2)
        steps_per_epoch = (len(train) - c.window_length + 1) // c.batch_size
        if (n != len(valid) or len(valid) < c.window_length
                or c.max_epoch * steps_per_epoch < c.valid_step_freq
                or not np.array_equal(values[:-n], train) or not np.array_equal(values[-n:], valid)):
            raise ValueError("canonical split or native validation eligibility failed before graph construction")
        self.build()
        import omni_anomaly.training as training
        original = training.TrainLoop
        worker = self
        trace, validations = [], []
        selection = {}

        class ObservedLoop(original):
            def _enter(loop):
                result = super(ObservedLoop, loop)._enter()
                selection["initial_model"] = worker.state()
                native_update = loop._early_stopping.update

                def update(metric, global_step=None):
                    if not np.isfinite(metric):
                        raise ValueError("nonfinite native validation monitor")
                    saved = native_update(metric, global_step)
                    row = {"epoch": loop.epoch, "step": global_step, "loss": float(metric), "saved": bool(saved)}
                    validations.append(row)
                    if saved:
                        selection.update(best_epoch=loop.epoch, best_step=global_step,
                                         best_loss=float(metric), selected_model=worker.state())
                    return saved
                loop._early_stopping.update = update
                return result

            def collect_metrics(loop, metrics=None, **kwargs):
                result = super(ObservedLoop, loop).collect_metrics(metrics, **kwargs)
                values = dict(metrics or {}, **kwargs)
                if "loss" in values:
                    value = float(values["loss"])
                    if not np.isfinite(value):
                        raise ValueError("nonfinite native training loss")
                    trace.append({"epoch": loop.epoch, "step": loop.step, "loss": value})
                return result

            def _exit(loop, *args):
                selection["terminal_model"] = worker.state()
                result = super(ObservedLoop, loop)._exit(*args)
                selection["restored_model"] = worker.state()
                return result

        training.TrainLoop = ObservedLoop
        try:
            native_result = self.trainer.fit(values, valid_portion=.2)
        finally:
            training.TrainLoop = original
        if not validations or selection["selected_model"] != selection["restored_model"]:
            raise ValueError("native best-state restoration failed")
        expected_steps = c.max_epoch * steps_per_epoch
        if len(trace) != expected_steps or int(self.session.run(self.trainer._global_step)) != expected_steps:
            raise ValueError("native training trajectory step count differs")
        best = min(validations, key=lambda row: row["loss"])
        if selection["best_step"] != best["step"] or native_result["best_valid_loss"] != best["loss"]:
            raise ValueError("native strict-first-tie selection differs")
        self.facts = {"training": trace, "validations": validations, "selection": selection,
                      "optimizer_steps": expected_steps, "training_calls": 1,
                      "native_result": {k: float(v) for k, v in native_result.items()},
                      "partitions": {"train": array_binding(train), "valid": array_binding(valid)},
                      "regularization_loss_count": len(self.tf.get_collection(self.tf.GraphKeys.REGULARIZATION_LOSSES))}
        self.trained = True
        # Native main scores the complete (train + validation) preprocessed
        # train-side series before test. Preserve the stochastic call order.
        self.predict(values, "train_prefix")
        return self.facts

    def discover_random(self, fetches):
        seen, pending = set(), [tensor.op for tensor in fetches]
        while pending:
            op = pending.pop()
            if op in seen:
                continue
            seen.add(op)
            pending.extend(t.op for t in op.inputs)
            pending.extend(op.control_inputs)
        # Enumerate every stateful random dependency, including random ops in
        # control-flow frames. Reject any newly introduced unhandled sampler.
        random_ops = sorted((op for op in seen if "Random" in op.type or op.type in ("TruncatedNormal", "Multinomial")), key=lambda op: op.name)
        if len(random_ops) != 1 or random_ops[0].type != "TruncatedNormal":
            raise ValueError("uncovered stochastic score dependencies: " + repr([(o.name, o.type) for o in random_ops]))
        self.noise = random_ops[0].outputs[0]
        self.random_identity = [{"name": op.name, "type": op.type, "seed": op.get_attr("seed"),
                                 "seed2": op.get_attr("seed2")} for op in random_ops]

    def predict(self, values, role):
        import omni_anomaly.prediction as prediction
        np = self.np
        if len(values) < self.config.window_length:
            raise ValueError("test input shorter than native window")
        occurrence = {"ordinal": len(self.occurrences), "role": role, "input": array_binding(values),
                      "first_batch": len(self.records)}
        if self.replaying:
            occurrence = self.replay_occurrence
            if occurrence["input"] != array_binding(values):
                raise ValueError("replay input or occurrence differs")
        worker = self
        batch_records = []

        class TraceSession:
            def run(self, requested, feed_dict):
                if not hasattr(worker, "noise"):
                    worker.discover_random(requested)
                batch = feed_dict[worker.predictor._input_x]
                ordinal = occurrence["first_batch"] + len(batch_records)
                record = {"batch": ordinal, "occurrence": occurrence["ordinal"],
                          "input": array_binding(batch), "noise_tensor": worker.noise.name}
                if worker.replaying:
                    ref = worker.saved["batches"][ordinal]
                    if any(record[k] != ref[k] for k in record):
                        raise ValueError("replay batch order/input binding differs")
                    draw = worker.saved_noise["batch_%d" % ordinal]
                    if array_binding(draw) != ref["noise"]:
                        raise ValueError("replay noise binding differs")
                    feeds = dict(feed_dict)
                    feeds[worker.noise] = draw
                    output = worker.session.run(requested, feed_dict=feeds)
                elif worker.reference:
                    output = worker.session.run(requested, feed_dict=feed_dict)
                else:
                    output = worker.session.run(list(requested) + [worker.noise], feed_dict=feed_dict)
                    draw = output.pop()
                    record["noise"] = array_binding(draw)
                    worker.noises["batch_%d" % ordinal] = draw
                batch_records.append(record)
                return output

        original = prediction.get_default_session_or_error
        prediction.get_default_session_or_error = lambda: TraceSession()
        try:
            score, _, _ = self.predictor.get_score(values)
        finally:
            prediction.get_default_session_or_error = original
        score = -score
        if score.shape != (len(values) - self.config.window_length + 1,) or not np.isfinite(score).all():
            raise ValueError("native predictor returned invalid scores")
        if self.replaying:
            if len(batch_records) != occurrence["batch_count"] or array_binding(score) != occurrence["score"]:
                raise ValueError("reloaded scores differ from frozen native scores")
        else:
            self.records.extend(batch_records)
            occurrence.update(batch_count=len(batch_records), score=array_binding(score))
            self.occurrences.append(occurrence)
        path = self.root / ("score-%d.npy" % occurrence["ordinal"])
        with path.open("xb") as stream:
            np.save(stream, score, allow_pickle=False)
        return {"path": path.name, **array_binding(score)}

    def finish(self):
        from tfsnippet.scaffold import VariableSaver
        if not self.trained or len(self.occurrences) < 2:
            raise ValueError("checkpoint requires native train-prefix and test scoring")
        bundle = self.root / "tensorflow"
        bundle.mkdir()
        VariableSaver(self.variables, str(bundle / "checkpoint")).save()
        graph = self.tf.get_default_graph()
        endpoints = {"input": self.predictor._input_x.name,
                     "score": self.predictor._score_without_y.name, "z": self.predictor._q_net_z.name}
        manifest = {"configuration": self.configuration, "runtime": self.runtime,
                    "random_dependencies": self.random_identity, "endpoints": endpoints,
                    "checkpoint_prefix": "checkpoint/variables.dat", "occurrences": self.occurrences,
                    "batches": self.records, "reference": self.reference,
                    "model_variables": {k: v.name for k, v in self.model_variables.items()},
                    "variable_inventory": {k: {"name": v.name, "shape": v.shape.as_list(), "dtype": v.dtype.name}
                                           for k, v in self.variables.items()},
                    "graph_sha256": digest(graph.as_graph_def().SerializeToString())}
        self.np.savez(str(bundle / "noise.npz"), **self.noises)
        write_json(bundle / "training.json", self.facts)
        write_json(bundle / "replay.json", manifest)
        return {"directory": str(bundle), "replay": manifest, "training": self.facts}

    def load(self, request):
        if self.session is not None or set(request) != {"action", "bundle"}:
            raise ValueError("reload requires a fresh worker")
        bundle = Path(request["bundle"])
        self.saved = json.loads((bundle / "replay.json").read_text())
        if self.saved["reference"] or self.saved["configuration"] != self.configuration or self.saved["runtime"] != self.runtime:
            raise ValueError("replay runtime/configuration/identity mismatch")
        prefix = bundle / self.saved["checkpoint_prefix"]
        if str(self.saved["checkpoint_prefix"]) != "checkpoint/variables.dat":
            raise ValueError("checkpoint prefix must be relative to bound bundle")
        # Restore the actual saved graph and explicit local prefix. Never follow
        # the native 'latest' file's old absolute path after directory migration.
        saver = self.tf.train.import_meta_graph(str(prefix) + ".meta")
        self.session_open()
        saver.restore(self.session, str(prefix))
        graph = self.tf.get_default_graph()
        from omni_anomaly.prediction import Predictor
        self.predictor = object.__new__(Predictor)
        self.predictor._model = SimpleNamespace(window_length=self.config.window_length, x_dims=1)
        self.predictor._batch_size = self.config.batch_size
        self.predictor._feed_dict = {}
        endpoints = self.saved["endpoints"]
        self.predictor._input_x = graph.get_tensor_by_name(endpoints["input"])
        self.predictor._score_without_y = graph.get_tensor_by_name(endpoints["score"])
        self.predictor._q_net_z = graph.get_tensor_by_name(endpoints["z"])
        self.discover_random(self.predictor._get_score_without_y())
        if self.random_identity != self.saved["random_dependencies"]:
            raise ValueError("restored random dependencies differ")
        self.model_variables = {key: graph.get_tensor_by_name(name) for key, name in self.saved["model_variables"].items()}
        facts = json.loads((bundle / "training.json").read_text())
        if self.state() != facts["selection"]["restored_model"]:
            raise ValueError("reloaded native selected variables differ")
        self.saved_noise = self.np.load(str(bundle / "noise.npz"), allow_pickle=False)
        self.replaying = True
        return {"restored": True, "training_calls": 0, "optimizer_steps": 0}

    def smoke(self):
        self.build()
        self.session.run(self.tf.global_variables_initializer())
        values = self.np.zeros((self.config.batch_size, self.config.window_length, 1), dtype=self.np.float32)
        loss, _ = self.session.run([self.trainer._loss, self.trainer._train_op],
                                  {self.trainer._input_x: values, self.trainer._learning_rate: self.config.initial_lr})
        if not self.np.isfinite(loss):
            raise ValueError("default graph smoke loss is nonfinite")
        return {"passed": True, "synthetic_optimizer_steps": 1, "real_training_calls": 0,
                "model_variables": len(self.model_variables), "global_variables": len(self.variables)}

    def handle(self, request):
        action = request["action"]
        if action == "train":
            return self.train(request)
        if action == "score" and set(request) == {"action", "values"} and self.trained:
            return self.predict(self.read_array(request["values"]), "test")
        if action == "finish" and set(request) == {"action"}:
            return self.finish()
        if action == "load":
            return self.load(request)
        if action == "replay" and set(request) == {"action", "values", "occurrence"} and self.replaying:
            self.replay_occurrence = self.saved["occurrences"][request["occurrence"]]
            return self.predict(self.read_array(request["values"]), self.replay_occurrence["role"])
        if action == "smoke" and set(request) == {"action"} and self.session is None:
            return self.smoke()
        raise ValueError("invalid worker action or state")


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Native prints never enter the JSON control stream.
    worker = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request == {"action": "close"}:
                break
            if worker is None:
                worker = initialize(request)
                result = {"runtime": worker.runtime}
            else:
                result = worker.handle(request)
            protocol.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            protocol.flush()
    finally:
        if worker is not None and worker.session is not None:
            worker.session.__exit__(None, None, None)


if __name__ == "__main__":
    main()
