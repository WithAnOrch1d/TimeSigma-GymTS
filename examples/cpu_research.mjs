/** CPU-only acceptance wrapper; the production Core12 Catalog is unchanged. */
import { pathToFileURL } from 'node:url';

export function cpuWorker(worker) {
  const allowed = new Set();
  return {
    async callSuccess(tool, payload, options, signal) {
      if (tool === 'experiment.run' && !allowed.has(payload.preview_id)) {
        throw new Error('CPU example requires a ready preview of an AR-only Pipeline without operators/ensemble');
      }
      const response = await worker.callSuccess(tool, payload, options, signal);
      if (tool === 'experiment.preview') {
        const result = response.result;
        const candidate = result?.candidate_pipeline;
        if (result?.status === 'ready' && candidate?.method === 'AR'
            && !candidate.ensemble_ref && Array.isArray(candidate.operators)
            && candidate.operators.length === 0 && typeof result.preview_id === 'string') {
          allowed.add(result.preview_id);
        }
      }
      return response;
    },
    stop: () => worker.stop(),
  };
}

export async function main(args) {
  if (!args.includes('--execute')) throw new Error('This calls a paid model; pass --execute explicitly');
  process.env.CUDA_VISIBLE_DEVICES = '-1';
  const { runResearchCli } = await import('../pi-host/dist/research-cli.js');
  const { runResearchFeedback } = await import('../pi-host/dist/research-feedback.js');
  const { PythonWorkerClient } = await import('../pi-host/dist/python-worker.js');
  return runResearchCli(args.filter(x => x !== '--execute'), {
    createWorker: options => cpuWorker(new PythonWorkerClient(options)),
    runFeedback: options => runResearchFeedback({...options,
      firstPrompt: 'This is a CPU-only integration acceptance, not an algorithm ranking experiment. '
        + 'Only AR without operators or ensemble is executable in this wrapper. The production Catalog remains Core12. '
        + 'Use the declared initial Component baseline and historical evidence. Propose one new, controlled AR '
        + 'hyperparameter or context-length hypothesis; preview it, then execute and diagnose its actual official metrics, '
        + 'and finish/commit the ResearchStep. Avoid exact existing Pipelines; do not invent findings. '
        + 'Do not change algorithm. You may finish without an experiment if no justified candidate exists.'}),
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exitCode = await main(process.argv.slice(2));
}
